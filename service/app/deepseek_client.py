"""DeepSeek 客户端：OpenAI 兼容的 /chat/completions 调用 + Anthropic 兼容的联网检索。

三种调用方式：
  chat_json            —— 强制 response_format=json_object，解析失败会把错误回喂给模型重试；
                          用于「备注症状抽取」与「建议生成」。
  chat_with_tools      —— 纯文本输出（不加 response_format），并带函数调用
                          （OpenAI tools / tool_calls）：用于没有地点的 /api/chat。
                          聊天回复应当是给人读的散文，套一层 JSON 只会让模型把精力
                          花在格式上。要不要查流行病学情报由模型自己决定，客户端只负责
                          搬运——真正执行工具的函数由流水线注入（tool_executor）。
  chat_anthropic_search —— **换一个协议**：DeepSeek 的联网检索只在 Anthropic 兼容端点
                          （<base>/anthropic/v1/messages）上提供，OpenAI 端点没有这个
                          服务端工具。用于 /api/destination 与「问题里出现了地点」的
                          /api/chat。返回 {"reply", "sources", "search_count"}。

这三条路共用一套超时/异常约定（失败一律抛 DeepSeekError），但**不共用协议**：
Anthropic 那条把 system 提到顶层、把回复拆成 content 块，检索结果作为
web_search_tool_result 块回来，真实链接就藏在这些块里。客户端只做搬运与解析，
「什么时候该检索」是流水线的产品决策，不在这里。

MOCK_MODE=true 时不发任何网络请求，直接返回可信假数据，便于本地演示与测试。
假建议按**风险档位**（low/medium/high）分三套，让演示能看出差异。

这份分档文案还有第二个身份：**兜底模板**。真实模型生成的建议若两次都没通过
输出校验（app.verifier），流水线就退回 fallback_advice() ——因此这里只有一份
文案，演示模式与线上兜底共用，不存在两套会各自漂移的文本。
"""

import copy
import json
import logging
from typing import Callable

import anyio
import httpx

from app.config import get_settings
from app.intel import INTEL_TOOL_NAME, find_location

logger = logging.getLogger(__name__)

# JSON 解析失败时的最大重试次数（不含首次请求）
_JSON_RETRIES = 2

# 带工具的对话最多来回几轮（一轮 = 一次模型调用 + 执行它要求的所有工具）
_DEFAULT_TOOL_ROUNDS = 2

# ---------- Anthropic 兼容端点（联网检索唯一可用的协议）----------

# 相对 DEEPSEEK_BASE_URL 的路径
ANTHROPIC_MESSAGES_PATH = "/anthropic/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# 服务端检索工具的类型标识（由服务端执行，不需要我们回传结果）
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
WEB_SEARCH_TOOL_NAME = "web_search"

# 检索调用的默认输出预算。**不要调小**：实测 700 tokens 会被 max_tokens 截断，
# 因为检索结果本身就占了约 13.9k 输入 token，模型需要足够篇幅把它们讲完。
SEARCH_MAX_TOKENS = 4000


class DeepSeekError(Exception):
    """DeepSeek 调用失败（网络错误、HTTP 错误或持续返回非法 JSON）。"""


# ---------- MOCK 数据 ----------

# 特征调用在 MOCK 模式下不做任何推断，交由确定性编码处理
_MOCK_FEATURES = {"infer": {}}

# 风险档位：三个模型中最高的等级（见 pipeline.overall_tier）
TIERS: tuple[str, ...] = ("low", "medium", "high")
_DEFAULT_TIER = "medium"
_DEFAULT_LANG = "zh-CN"

# ---- 与风险档位无关的部分：防蚊防护与居家监测 ----

_MOCK_PROTECTION: dict[str, list[str]] = {
    "zh-CN": [
        "清除住所周边的积水容器（花盆托盘、水桶、废旧轮胎），从源头减少蚊虫孳生。",
        "使用纱窗、蚊帐，白天也要防蚊——传播登革热的伊蚊主要在日间叮咬。",
        "外出时穿浅色长袖衣裤，在裸露皮肤涂抹驱蚊剂。",
    ],
    "zh-TW": [
        "清除住所周邊的積水容器（花盆底盤、水桶、廢輪胎），從源頭減少病媒蚊孳生。",
        "使用紗窗、蚊帳，白天也要防蚊——傳播登革熱的斑蚊主要在日間叮咬。",
        "外出時穿淺色長袖衣褲，在裸露皮膚塗抹防蚊液。",
    ],
    "en": [
        "Remove standing water around your home (plant saucers, buckets, old tyres) to stop mosquitoes breeding.",
        "Use window screens and bed nets, and protect yourself during the day — the Aedes mosquitoes that spread dengue bite mainly in daylight.",
        "Wear light-coloured long sleeves and trousers outdoors, and apply repellent to exposed skin.",
    ],
    "es": [
        "Elimine los recipientes con agua estancada alrededor de su vivienda (platos de macetas, baldes, neumáticos) para cortar la cría de mosquitos.",
        "Use mosquiteros en ventanas y camas, y protéjase también de día: el mosquito Aedes que transmite el dengue pica principalmente durante el día.",
        "Al salir, use ropa clara de manga larga y aplique repelente en la piel expuesta.",
    ],
    "pt": [
        "Elimine recipientes com água parada ao redor de casa (pratos de vasos, baldes, pneus velhos) para impedir a criação do mosquito.",
        "Use telas nas janelas e mosquiteiro, e proteja-se também durante o dia — o Aedes, transmissor da dengue, pica principalmente à luz do dia.",
        "Ao sair, use roupas claras de manga comprida e aplique repelente na pele exposta.",
    ],
}

_MOCK_MONITORING: dict[str, list[str]] = {
    "zh-CN": [
        "每日早晚测量体温并记录，注意退热后 24-48 小时是病情变化的关键窗口。",
        "保证充足饮水，留意尿量是否明显减少。",
        "观察皮肤有无新出现的出血点或瘀斑。",
    ],
    "zh-TW": [
        "每日早晚量測體溫並記錄，注意退燒後 24-48 小時是病情變化的關鍵時期。",
        "確保充足飲水，留意尿量是否明顯減少。",
        "觀察皮膚有無新出現的出血點或瘀斑。",
    ],
    "en": [
        "Record your temperature every morning and evening; the 24–48 hours after fever subsides is the critical window.",
        "Keep well hydrated and watch for a marked drop in urine output.",
        "Check your skin for new bleeding spots or bruising.",
    ],
    "es": [
        "Mida y registre su temperatura por la mañana y por la noche; las 24–48 horas posteriores a la caída de la fiebre son el periodo crítico.",
        "Manténgase bien hidratado y observe si disminuye notablemente la cantidad de orina.",
        "Revise su piel en busca de nuevos puntos de sangrado o moretones.",
    ],
    "pt": [
        "Meça e registre a temperatura de manhã e à noite; as 24–48 horas após a queda da febre são o período crítico.",
        "Mantenha-se bem hidratado e observe se o volume de urina diminui de forma acentuada.",
        "Verifique a pele em busca de novos pontos de sangramento ou manchas roxas.",
    ],
}

# ---- 随风险档位变化的部分：就医提示与总结 ----
# 演示时最该看出差别的就是这两块：低风险不该催人去医院，
# 高风险必须明确写出「尽快就医」。

_MOCK_MEDICAL: dict[str, dict[str, list[str]]] = {
    "zh-CN": {
        "low": [
            "目前无需急诊处理；如果发热持续超过 48 小时或出现新的症状，请到医疗机构就诊。",
            "退热镇痛请遵医嘱，避免自行服用阿司匹林或布洛芬类药物，以免增加出血风险。",
        ],
        "medium": [
            "建议近期就医评估，并向医生说明症状出现的时间与变化过程。",
            "如出现持续呕吐、剧烈腹痛、牙龈或鼻腔出血、嗜睡或烦躁不安，请立即就医。",
            "退热镇痛请遵医嘱，避免自行服用阿司匹林或布洛芬类药物，以免增加出血风险。",
        ],
        "high": [
            "请尽快就医：本次评估提示风险相对偏高，需要由医生当面评估并考虑血常规等检查。",
            "若出现持续呕吐、剧烈腹痛、黏膜出血、嗜睡或意识改变，请立即前往急诊。",
            "就医前不要自行服用阿司匹林或布洛芬类药物，以免增加出血风险。",
        ],
    },
    "zh-TW": {
        "low": [
            "目前無需急診處理；若發燒持續超過 48 小時或出現新的症狀，請至醫療院所就診。",
            "退燒止痛請遵醫囑，避免自行服用阿斯匹靈或布洛芬類藥物，以免增加出血風險。",
        ],
        "medium": [
            "建議近期就醫評估，並向醫師說明症狀出現的時間與變化過程。",
            "如出現持續嘔吐、劇烈腹痛、牙齦或鼻腔出血、嗜睡或躁動不安，請立即就醫。",
            "退燒止痛請遵醫囑，避免自行服用阿斯匹靈或布洛芬類藥物，以免增加出血風險。",
        ],
        "high": [
            "請盡快就醫：本次評估顯示風險相對偏高，需要由醫師當面評估並考慮血液檢查。",
            "若出現持續嘔吐、劇烈腹痛、黏膜出血、嗜睡或意識改變，請立即前往急診。",
            "就醫前請勿自行服用阿斯匹靈或布洛芬類藥物，以免增加出血風險。",
        ],
    },
    "en": {
        "low": [
            "No urgent care is needed right now. See a clinician if the fever lasts beyond 48 hours or new symptoms appear.",
            "Take fever or pain medication only as advised by a clinician, and avoid self-medicating with aspirin or ibuprofen, which can increase bleeding risk.",
        ],
        "medium": [
            "Arrange a medical review in the next day or two, and tell the clinician when your symptoms started and how they have changed.",
            "Seek care immediately if you develop persistent vomiting, severe abdominal pain, bleeding gums or nose, drowsiness, or restlessness.",
            "Take fever or pain medication only as advised by a clinician, and avoid self-medicating with aspirin or ibuprofen, which can increase bleeding risk.",
        ],
        "high": [
            "Seek medical care promptly: this assessment places you relatively high, and a clinician should examine you and consider blood tests.",
            "Go to an emergency department immediately if you develop persistent vomiting, severe abdominal pain, bleeding, drowsiness, or confusion.",
            "Do not take aspirin or ibuprofen before you are seen — they can increase bleeding risk.",
        ],
    },
    "es": {
        "low": [
            "Por ahora no se requiere atención urgente. Consulte si la fiebre dura más de 48 horas o aparecen síntomas nuevos.",
            "Tome analgésicos o antipiréticos solo bajo indicación médica y evite automedicarse con aspirina o ibuprofeno, que pueden aumentar el riesgo de sangrado.",
        ],
        "medium": [
            "Programe una consulta médica en los próximos días e indique cuándo comenzaron sus síntomas y cómo han evolucionado.",
            "Acuda de inmediato a un centro de salud si presenta vómitos persistentes, dolor abdominal intenso, sangrado de encías o nariz, somnolencia o inquietud.",
            "Tome analgésicos o antipiréticos solo bajo indicación médica y evite automedicarse con aspirina o ibuprofeno, que pueden aumentar el riesgo de sangrado.",
        ],
        "high": [
            "Busque atención médica lo antes posible: esta evaluación lo sitúa en una posición relativamente alta y conviene que un profesional lo examine y valore análisis de sangre.",
            "Acuda de urgencia si presenta vómitos persistentes, dolor abdominal intenso, sangrado, somnolencia o confusión.",
            "No tome aspirina ni ibuprofeno antes de ser evaluado: pueden aumentar el riesgo de sangrado.",
        ],
    },
    "pt": {
        "low": [
            "Não é necessário atendimento de urgência agora. Procure um profissional se a febre passar de 48 horas ou surgirem sintomas novos.",
            "Use antitérmicos ou analgésicos apenas com orientação médica e evite automedicação com aspirina ou ibuprofeno, que podem aumentar o risco de sangramento.",
        ],
        "medium": [
            "Agende uma avaliação médica nos próximos dias e informe quando os sintomas começaram e como evoluíram.",
            "Procure atendimento imediatamente se surgirem vômitos persistentes, dor abdominal intensa, sangramento de gengiva ou nariz, sonolência ou agitação.",
            "Use antitérmicos ou analgésicos apenas com orientação médica e evite automedicação com aspirina ou ibuprofeno, que podem aumentar o risco de sangramento.",
        ],
        "high": [
            "Procure atendimento médico o quanto antes: esta avaliação indica risco relativamente alto e um profissional deve examiná-lo e considerar exames de sangue.",
            "Vá ao pronto-socorro imediatamente se surgirem vômitos persistentes, dor abdominal intensa, sangramento, sonolência ou confusão.",
            "Não tome aspirina nem ibuprofeno antes da avaliação — podem aumentar o risco de sangramento.",
        ],
    },
}

_MOCK_SUMMARY: dict[str, dict[str, str]] = {
    "zh-CN": {
        "low": "根据您填写的症状与病史，当前登革热相关风险处于相对较低水平，请继续做好防蚊防护并留意症状变化。",
        "medium": "根据您填写的症状与病史，当前登革热相关风险处于中等水平，建议加强防蚊并密切观察病情变化。",
        "high": "根据您填写的症状与病史，当前登革热相关风险相对偏高，建议尽快就医评估，同时持续做好防蚊与病情监测。",
    },
    "zh-TW": {
        "low": "根據您填寫的症狀與病史，目前登革熱相關風險處於相對較低水準，請持續做好防蚊防護並留意症狀變化。",
        "medium": "根據您填寫的症狀與病史，目前登革熱相關風險處於中等水準，建議加強防蚊並密切觀察病情變化。",
        "high": "根據您填寫的症狀與病史，目前登革熱相關風險相對偏高，建議盡快就醫評估，同時持續做好防蚊與病情監測。",
    },
    "en": {
        "low": (
            "Based on the symptoms and history you provided, your dengue-related risk appears "
            "relatively low. Keep up mosquito protection and watch for any change in symptoms."
        ),
        "medium": (
            "Based on the symptoms and history you provided, your dengue-related risk appears "
            "moderate. Strengthen mosquito protection and monitor your condition closely."
        ),
        "high": (
            "Based on the symptoms and history you provided, your dengue-related risk appears "
            "relatively high. Seek medical assessment promptly and keep monitoring your condition."
        ),
    },
    "es": {
        "low": (
            "Según los síntomas y antecedentes que indicó, su riesgo relacionado con el dengue "
            "es relativamente bajo. Mantenga la protección contra mosquitos y vigile cualquier cambio."
        ),
        "medium": (
            "Según los síntomas y antecedentes que indicó, su riesgo relacionado con el dengue "
            "es moderado. Refuerce la protección contra mosquitos y vigile de cerca su evolución."
        ),
        "high": (
            "Según los síntomas y antecedentes que indicó, su riesgo relacionado con el dengue "
            "es relativamente alto. Busque atención médica lo antes posible y siga vigilando su evolución."
        ),
    },
    "pt": {
        "low": (
            "Com base nos sintomas e antecedentes informados, seu risco relacionado à dengue "
            "está relativamente baixo. Mantenha a proteção contra mosquitos e observe mudanças."
        ),
        "medium": (
            "Com base nos sintomas e antecedentes informados, seu risco relacionado à dengue "
            "está em nível moderado. Reforce a proteção contra mosquitos e acompanhe de perto sua evolução."
        ),
        "high": (
            "Com base nos sintomas e antecedentes informados, seu risco relacionado à dengue "
            "está relativamente alto. Procure avaliação médica o quanto antes e siga monitorando sua evolução."
        ),
    },
}

# ---- /api/chat 的假回复：引用用户自己的风险等级，措辞保守 ----

_MOCK_CHAT_TIER_LABELS: dict[str, dict[str, str]] = {
    "zh-CN": {"low": "较低", "medium": "中等", "high": "偏高"},
    "zh-TW": {"low": "較低", "medium": "中等", "high": "偏高"},
    "en": {"low": "low", "medium": "moderate", "high": "high"},
    "es": {"low": "bajo", "medium": "moderado", "high": "alto"},
    "pt": {"low": "baixo", "medium": "moderado", "high": "alto"},
}

_MOCK_CHAT_TEMPLATES: dict[str, str] = {
    "zh-CN": (
        "（演示模式回复，未调用真实模型）您本次评估的总体风险为「{tier}」。"
        "需要说明的是，评分只是相对参考值，并不代表感染概率，也不能替代医生的判断。"
        "请继续做好防蚊防护、充分补液与休息；若症状加重，或出现持续呕吐、剧烈腹痛、"
        "出血表现、嗜睡等警示征象，请尽快就医。"
    ),
    "zh-TW": (
        "（示範模式回覆，未呼叫真實模型）您本次評估的整體風險為「{tier}」。"
        "需要說明的是，評分只是相對參考值，並不代表感染機率，也無法取代醫師的判斷。"
        "請持續做好防蚊防護、充分補水與休息；若症狀加重，或出現持續嘔吐、劇烈腹痛、"
        "出血表現、嗜睡等警示徵象，請盡快就醫。"
    ),
    "en": (
        "(Demo-mode reply — no live model was called.) Your overall risk on this assessment is "
        "{tier}. Remember that the scores are relative indicators, not a probability of infection, "
        "and they cannot replace a clinician's judgement. Keep up mosquito protection, fluids and "
        "rest; if your symptoms worsen, or you develop persistent vomiting, severe abdominal pain, "
        "bleeding or drowsiness, seek medical care promptly."
    ),
    "es": (
        "(Respuesta en modo demostración: no se llamó a ningún modelo real.) Su riesgo general en "
        "esta evaluación es {tier}. Recuerde que las puntuaciones son indicadores relativos, no una "
        "probabilidad de infección, y no sustituyen el criterio de un profesional. Mantenga la "
        "protección contra mosquitos, la hidratación y el reposo; si sus síntomas empeoran o "
        "aparecen vómitos persistentes, dolor abdominal intenso, sangrado o somnolencia, busque "
        "atención médica sin demora."
    ),
    "pt": (
        "(Resposta em modo de demonstração — nenhum modelo real foi chamado.) Seu risco geral "
        "nesta avaliação é {tier}. Lembre-se de que as pontuações são indicadores relativos, não "
        "uma probabilidade de infecção, e não substituem a avaliação de um profissional. Mantenha "
        "a proteção contra mosquitos, a hidratação e o repouso; se os sintomas piorarem ou "
        "surgirem vômitos persistentes, dor abdominal intensa, sangramento ou sonolência, procure "
        "atendimento médico rapidamente."
    ),
}


def _normalise(language: str, tier: str) -> tuple[str, str]:
    """把语言与档位收敛到有假数据的取值上。"""
    lang = language if language in _MOCK_PROTECTION else _DEFAULT_LANG
    level = tier if tier in TIERS else _DEFAULT_TIER
    return lang, level


def fallback_advice(language: str, tier: str) -> dict:
    """按语言与风险档位组装一份**可直接返回给用户**的建议。

    两个调用方共用这一份文案，刻意不复制：
      - MOCK_MODE 下 chat_json("advice") 的返回值；
      - 真实模式下模型失败、或连续两次没通过 app.verifier 校验后的兜底。
    键顺序与 Advice 模型一致（medical → monitoring → protection）。
    """
    lang, level = _normalise(language, tier)
    return {
        "summary": _MOCK_SUMMARY[lang][level],
        "advice": {
            "medical": list(_MOCK_MEDICAL[lang][level]),
            "monitoring": list(_MOCK_MONITORING[lang]),
            "protection": list(_MOCK_PROTECTION[lang]),
        },
    }


# 旧名保留：演示模式下它就是兜底模板，同一个函数
build_mock_advice = fallback_advice


# ---- 出行前建议：与「已经生病了」那套刻意分开的一份文案 ----
#
# /api/destination 的用户还没生病，也没有评分。把 _MOCK_MEDICAL 那套
# 「您本次评估的风险为…」直接端过去会答非所问，所以就医与监测另写一份，
# 只有防蚊那部分与评估路径共用（防蚊建议对病人和旅客是同一件事）。
# 分两档：endemic（high / moderate）与 quiet（low / none / unknown）——
# 再细分就要开始编数字了，那正是这个服务不做的事。
_ENDEMIC_LEVELS: tuple[str, ...] = ("high", "moderate")

_TRAVEL_MEDICAL: dict[str, dict[str, list[str]]] = {
    "zh-CN": {
        "endemic": [
            "这是登革热流行地区：出发前先了解当地的就医渠道，随身带好保险与联系方式。",
            "旅途中或回程后两周内出现发热，请尽快就医，并主动告知医生这段旅行史。",
        ],
        "quiet": [
            "该地区目前没有持续的本地传播记录，无需为登革热做特别的医疗准备。",
            "若旅途中或回程后两周内出现发热，仍请就医并说明旅行史——发热的原因不止一种。",
        ],
    },
    "zh-TW": {
        "endemic": [
            "這是登革熱流行地區：出發前先了解當地的就醫管道，隨身帶好保險與聯絡方式。",
            "旅途中或回程後兩週內出現發燒，請儘快就醫，並主動告知醫師這段旅遊史。",
        ],
        "quiet": [
            "該地區目前沒有持續的本地傳播紀錄，不需為登革熱做特別的醫療準備。",
            "若旅途中或回程後兩週內出現發燒，仍請就醫並說明旅遊史——發燒的原因不只一種。",
        ],
    },
    "en": {
        "endemic": [
            "This is a dengue-endemic area: before you travel, find out how to reach medical care "
            "there and keep your insurance details with you.",
            "If you develop a fever during the trip or within two weeks of returning, seek medical "
            "care and tell the clinician where you have been.",
        ],
        "quiet": [
            "There is no record of sustained local transmission here, so no dengue-specific medical "
            "preparation is needed for the trip.",
            "If you develop a fever during the trip or within two weeks of returning, still see a "
            "clinician and mention the travel — fever has more than one cause.",
        ],
    },
    "es": {
        "endemic": [
            "Es una zona endémica de dengue: antes del viaje averigüe cómo acceder a atención médica "
            "allí y lleve consigo los datos de su seguro.",
            "Si aparece fiebre durante el viaje o en las dos semanas posteriores, busque atención "
            "médica e informe al personal de que estuvo allí.",
        ],
        "quiet": [
            "No hay registro de transmisión local sostenida en la zona, con lo que no se requiere "
            "ninguna preparación médica específica para el dengue.",
            "Si aparece fiebre durante el viaje o en las dos semanas posteriores, consulte igualmente "
            "e indique el viaje: la fiebre tiene más de una causa.",
        ],
    },
    "pt": {
        "endemic": [
            "Esta é uma área endêmica de dengue: antes da viagem, descubra como chegar ao atendimento "
            "médico local e leve os dados do seu seguro.",
            "Se surgir febre durante a viagem ou nas duas semanas seguintes ao retorno, procure "
            "atendimento médico e informe onde esteve.",
        ],
        "quiet": [
            "Não há registro de transmissão local sustentada nessa área, portanto não é preciso "
            "nenhuma preparação médica específica para dengue.",
            "Se surgir febre durante a viagem ou nas duas semanas seguintes, procure um profissional "
            "mesmo assim e informe a viagem — febre tem mais de uma causa.",
        ],
    },
}

_TRAVEL_MONITORING: dict[str, list[str]] = {
    "zh-CN": [
        "旅途中每天留意有没有发热、头痛、眼后痛或皮疹。",
        "回程后两周内继续关注体温——这段时间正好覆盖登革热的潜伏期。",
        "一旦出现持续呕吐、剧烈腹痛、出血表现或嗜睡，请立即就医。",
    ],
    "zh-TW": [
        "旅途中每天留意有沒有發燒、頭痛、眼後痛或皮疹。",
        "回程後兩週內持續關注體溫——這段時間正好涵蓋登革熱的潛伏期。",
        "一旦出現持續嘔吐、劇烈腹痛、出血表現或嗜睡，請立即就醫。",
    ],
    "en": [
        "While you are there, watch each day for fever, headache, pain behind the eyes or a rash.",
        "Keep checking your temperature for two weeks after you get back — that window covers the "
        "incubation period.",
        "Persistent vomiting, severe abdominal pain, bleeding or drowsiness means going to a doctor "
        "straight away.",
    ],
    "es": [
        "Durante la estancia, vigile cada día si aparecen fiebre, dolor de cabeza, dolor detrás de "
        "los ojos o erupción.",
        "Siga controlando la temperatura las dos semanas posteriores al regreso: ese margen cubre el "
        "periodo de incubación.",
        "Ante vómitos persistentes, dolor abdominal intenso, sangrado o somnolencia, acuda de "
        "inmediato a un centro de salud.",
    ],
    "pt": [
        "Durante a estadia, observe todos os dias se surgem febre, dor de cabeça, dor atrás dos olhos "
        "ou manchas na pele.",
        "Continue medindo a temperatura nas duas semanas após o retorno — esse intervalo cobre o "
        "período de incubação.",
        "Vômitos persistentes, dor abdominal intensa, sangramento ou sonolência pedem atendimento "
        "imediato.",
    ],
}


def fallback_travel_advice(language: str, endemicity: str) -> dict:
    """出行前建议：{"protection": [...], "medical": [...], "monitoring": [...]}。

    与 fallback_advice 一样，这是**唯一一份**文案：/api/destination 无论检索成功
    与否都用它，演示模式与线上完全相同。它不来自模型，因此永远可用、永远合规。
    键顺序与 schemas.DestinationAdvice 一致（protection 在前）。
    """
    lang = language if language in _MOCK_PROTECTION else _DEFAULT_LANG
    bucket = "endemic" if endemicity in _ENDEMIC_LEVELS else "quiet"
    return {
        "protection": list(_MOCK_PROTECTION[lang]),
        "medical": list(_TRAVEL_MEDICAL[lang][bucket]),
        "monitoring": list(_TRAVEL_MONITORING[lang]),
    }


def build_mock_chat_reply(language: str, tier: str) -> str:
    """按语言与风险档位组装假聊天回复（引用用户自己的风险等级）。"""
    lang, level = _normalise(language, tier)
    return _MOCK_CHAT_TEMPLATES[lang].format(tier=_MOCK_CHAT_TIER_LABELS[lang][level])


# ---- MOCK 的「带工具」回复：真的走一遍工具，再引用它真的返回的那条链接 ----

_MOCK_ENDEMICITY_LABELS: dict[str, dict[str, str]] = {
    "zh-CN": {
        "high": "登革热高度流行地区", "moderate": "存在局部传播的地区",
        "low": "偶有本地传播的低风险地区", "none": "目前没有本地传播的地区",
        "unknown": "资料表中没有收录的地区",
    },
    "zh-TW": {
        "high": "登革熱高度流行地區", "moderate": "存在局部傳播的地區",
        "low": "偶有本地傳播的低風險地區", "none": "目前沒有本地傳播的地區",
        "unknown": "資料表中沒有收錄的地區",
    },
    "en": {
        "high": "a highly endemic area for dengue", "moderate": "an area with limited local transmission",
        "low": "a low-risk area with only occasional local transmission",
        "none": "an area with no established local transmission",
        "unknown": "an area this reference table does not cover",
    },
    "es": {
        "high": "una zona de alta endemicidad de dengue", "moderate": "una zona con transmisión local limitada",
        "low": "una zona de bajo riesgo con transmisión local ocasional",
        "none": "una zona sin transmisión local establecida",
        "unknown": "una zona que esta tabla de referencia no cubre",
    },
    "pt": {
        "high": "uma área de alta endemicidade de dengue", "moderate": "uma área com transmissão local limitada",
        "low": "uma área de baixo risco, com transmissão local ocasional",
        "none": "uma área sem transmissão local estabelecida",
        "unknown": "uma área que esta tabela de referência não cobre",
    },
}

_MOCK_TOOL_TEMPLATES: dict[str, str] = {
    "zh-CN": (
        "（演示模式回复，未调用真实模型）我查了内置的地区背景表与 WHO 疾病暴发新闻："
        "{location} 属于{label}。{season}\n"
        "出行或居住期间请做好防蚊防护：使用驱蚊剂与蚊帐、清除住所周边积水；"
        "白天同样要防蚊。若出现发热、头痛、眼后痛等症状，请尽快就医并主动告知旅行史。{cite}"
    ),
    "zh-TW": (
        "（示範模式回覆，未呼叫真實模型）我查了內建的地區背景表與 WHO 疾病暴發新聞："
        "{location} 屬於{label}。{season}\n"
        "出行或居住期間請做好防蚊防護：使用防蚊液與蚊帳、清除住所周邊積水；"
        "白天同樣要防蚊。若出現發燒、頭痛、眼後痛等症狀，請儘快就醫並主動告知旅遊史。{cite}"
    ),
    "en": (
        "(Demo-mode reply — no live model was called.) I checked the built-in country table and the "
        "WHO Disease Outbreak News: {location} is {label}. {season}\n"
        "While you are there, keep up mosquito protection — repellent, bed nets, and clearing standing "
        "water around where you stay — and remember the Aedes mosquito bites during the day. If you "
        "develop fever, headache or pain behind the eyes, seek medical care promptly and mention your "
        "travel history.{cite}"
    ),
    "es": (
        "(Respuesta en modo demostración: no se llamó a ningún modelo real.) Consulté la tabla interna "
        "de países y las noticias de brotes de la OMS: {location} es {label}. {season}\n"
        "Mientras esté allí, mantenga la protección contra mosquitos —repelente, mosquiteros y "
        "eliminación de agua estancada— y recuerde que el mosquito Aedes pica de día. Si aparece "
        "fiebre, dolor de cabeza o dolor detrás de los ojos, busque atención médica lo antes posible "
        "e informe de su viaje.{cite}"
    ),
    "pt": (
        "(Resposta em modo de demonstração — nenhum modelo real foi chamado.) Consultei a tabela "
        "interna de países e as notícias de surtos da OMS: {location} é {label}. {season}\n"
        "Enquanto estiver lá, mantenha a proteção contra mosquitos — repelente, mosquiteiro e remoção "
        "de água parada — e lembre-se de que o Aedes pica durante o dia. Se surgirem febre, dor de "
        "cabeça ou dor atrás dos olhos, procure atendimento médico o quanto antes e informe sua viagem.{cite}"
    ),
}

# 没有任何来源时的说明句：**明说查不到**，而不是悄悄不提
_MOCK_NO_SOURCE: dict[str, str] = {
    "zh-CN": "\n本次没有取到可引用的 WHO 通报，因此不提供链接。",
    "zh-TW": "\n本次未取得可引用的 WHO 通報，因此不提供連結。",
    "en": "\nNo citable WHO notice came back this time, so no link is given.",
    "es": "\nEsta vez no se obtuvo ningún aviso citable de la OMS, así que no se incluye enlace.",
    "pt": "\nDesta vez não foi obtido nenhum aviso citável da OMS, portanto nenhum link é fornecido.",
}

_MOCK_CITE: dict[str, str] = {
    "zh-CN": "\n参考来源（WHO 疾病暴发新闻）：{title}（{date}）{url}",
    "zh-TW": "\n參考來源（WHO 疾病暴發新聞）：{title}（{date}）{url}",
    "en": "\nSource (WHO Disease Outbreak News): {title} ({date}) {url}",
    "es": "\nFuente (Noticias sobre brotes de enfermedades, OMS): {title} ({date}) {url}",
    "pt": "\nFonte (Notícias sobre surtos de doenças, OMS): {title} ({date}) {url}",
}


def build_mock_tool_reply(language: str, result: dict) -> str:
    """用**工具真正返回的数据**拼一条演示回复，引用的链接必然来自 result。"""
    lang = language if language in _MOCK_TOOL_TEMPLATES else _DEFAULT_LANG
    endemicity = str(result.get("endemicity") or "unknown")
    labels = _MOCK_ENDEMICITY_LABELS[lang]
    season = result.get("season_note") or ""
    notices = result.get("who_notices") or []
    if notices:
        cite = _MOCK_CITE[lang].format(**notices[0])
    else:
        cite = _MOCK_NO_SOURCE[lang]
    return _MOCK_TOOL_TEMPLATES[lang].format(
        location=result.get("location", "?"),
        label=labels.get(endemicity, labels["unknown"]),
        season=season,
        cite=cite,
    )


# ---------- MOCK 的「联网检索」：固定的真实公共卫生页面，不是编出来的链接 ----------
#
# 这三条都是**真实存在且长期稳定**的页面，和真实检索会返回的东西同一类：
# 一条 WHO 疾病暴发新闻条目、两个区域监测入口。刻意不编造
# "https://health.gov.xx/dengue-2026-report" 这种看起来像真的的假链接——
# 演示模式给出的引用如果点不开，「不许编造来源」这条不变量在演示里就是假的。
# 第二条故意没有日期：真实检索经常拿不到 page_age，date=None 的分支必须被走到。
MOCK_SEARCH_SOURCES: tuple[dict, ...] = (
    {
        "title": "Dengue - Global situation (WHO Disease Outbreak News)",
        "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON518",
        "date": "2024-05-30",
    },
    {
        "title": "Dengue - PAHO/WHO",
        "url": "https://www.paho.org/en/topics/dengue",
        "date": None,
    },
    {
        "title": "Dengue worldwide overview - ECDC",
        "url": "https://www.ecdc.europa.eu/en/dengue-monthly",
        "date": "2026-07-31",
    },
)

# 演示模式的检索回复：**明说这是演示**、明说时间窗是最近三个月、只引用上面那份清单里的链接。
# 末尾的 {cite} 由 build_mock_search_reply 填成第一条来源的 URL。
_MOCK_SEARCH_TEMPLATES: dict[str, str] = {
    "zh-CN": (
        "（演示模式回复，未真正联网检索）以下是关于 {location} 最近三个月登革热情况的示例整理：\n"
        "- 公共卫生机构的登革热监测页面持续更新，需要以官方最新数据为准。\n"
        "- 传播强度随季节与降雨变化，出行前后都应做好防蚊防护。\n"
        "- 演示模式下没有真实检索结果，因此不给出具体病例数字。\n"
        "参考来源：{cite}"
    ),
    "zh-TW": (
        "（示範模式回覆，未真正連網檢索）以下是關於 {location} 最近三個月登革熱情況的範例整理：\n"
        "- 公共衛生機構的登革熱監測頁面持續更新，需以官方最新資料為準。\n"
        "- 傳播強度隨季節與降雨變化，出行前後都應做好防蚊防護。\n"
        "- 示範模式下沒有真實檢索結果，因此不提供具體病例數字。\n"
        "參考來源：{cite}"
    ),
    "en": (
        "(Demo-mode reply — no live web search was run.) Here is a sample summary of the dengue "
        "situation in {location} over the last three months:\n"
        "- The public-health surveillance pages for the area are updated regularly, and the official "
        "figures there are the ones that count.\n"
        "- Transmission follows the rainy season, so protection against mosquito bites matters "
        "throughout the trip.\n"
        "- No real search ran in demo mode, so no case counts are quoted here.\n"
        "Source: {cite}"
    ),
    "es": (
        "(Respuesta en modo demostración: no se realizó una búsqueda web real.) Este es un resumen de "
        "ejemplo sobre la situación del dengue en {location} en los últimos tres meses:\n"
        "- Las páginas de vigilancia de salud pública se actualizan con frecuencia y son las cifras "
        "oficiales las que valen.\n"
        "- La transmisión sigue a la temporada de lluvias, con lo que conviene protegerse de las "
        "picaduras durante todo el viaje.\n"
        "- En modo demostración no se ejecutó ninguna búsqueda, por eso no se citan casos.\n"
        "Fuente: {cite}"
    ),
    "pt": (
        "(Resposta em modo de demonstração — nenhuma busca real foi feita.) Este é um resumo de "
        "exemplo sobre a situação da dengue em {location} nos últimos três meses:\n"
        "- As páginas de vigilância em saúde pública são atualizadas com frequência, e os números "
        "oficiais são os que valem.\n"
        "- A transmissão acompanha o período de chuvas, então a proteção contra picadas importa "
        "durante toda a viagem.\n"
        "- Nenhuma busca real foi executada, portanto não há contagem de casos aqui.\n"
        "Fonte: {cite}"
    ),
}


# 检索一无所获时的说明句：**明说查不到**，不要用泛泛之谈填满这一段
_MOCK_SEARCH_EMPTY: dict[str, str] = {
    "zh-CN": "（演示模式回复）没有检索到这个地区最近三个月的登革热资料，因此不提供任何结论与链接。",
    "zh-TW": "（示範模式回覆）沒有檢索到這個地區最近三個月的登革熱資料，因此不提供任何結論與連結。",
    "en": (
        "(Demo-mode reply.) No recent information about the dengue situation there came back for the "
        "last three months, and so no findings or links are given."
    ),
    "es": (
        "(Respuesta en modo demostración.) No se encontró información sobre la situación del dengue "
        "allí en los últimos tres meses, con lo que no se dan conclusiones ni enlaces."
    ),
    "pt": (
        "(Resposta em modo de demonstração.) Não foi encontrada informação sobre a dengue nesse local "
        "nos últimos três meses, e por isso não há conclusões nem links."
    ),
}


def build_mock_search(language: str, location: str) -> dict:
    """演示模式下的一次「联网检索」。形状与真实解析结果逐字段一致。

    location 为空时表示上游没能认出任何地点：这时**不给来源**，回复也直说查不到——
    与真实检索一无所获时该走的那条路完全相同，演示因此能覆盖 degraded 分支。
    """
    lang = language if language in _MOCK_SEARCH_TEMPLATES else _DEFAULT_LANG
    place = (location or "").strip()
    if not place:
        return {"reply": _MOCK_SEARCH_EMPTY[lang], "sources": [], "search_count": 0}
    sources = [dict(s) for s in MOCK_SEARCH_SOURCES]
    reply = _MOCK_SEARCH_TEMPLATES[lang].format(location=place, cite=sources[0]["url"])
    return {"reply": reply, "sources": sources, "search_count": 1}


# ---------- Anthropic 协议：消息转换与响应解析 ----------


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """把 OpenAI 风格的消息列表转成 Anthropic 的 messages。

    两处差异：system 不是一条消息（由顶层 system 参数承载，这里直接丢弃），
    content 只保留字符串。空内容的消息会被跳过——Anthropic 端点拒绝空文本块。
    """
    converted: list[dict] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "system":  # system 走顶层参数，不能混在 messages 里
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if not (text or "").strip():
            continue
        converted.append(
            {"role": "assistant" if role == "assistant" else "user", "content": text}
        )
    return converted


def _harvest_sources(node, sources: list[dict], seen: set[str]) -> None:
    """在一个 web_search_tool_result 块里递归找出所有 url。

    为什么递归：真实返回里 url 埋在 content -> [ {type: web_search_result, url, title,
    page_age, encrypted_content} ] 这样的嵌套结构里，而这个结构没有稳定契约。
    只认「带 http(s) url 的对象」，顺带取同级的 title / page_age，其余一律不碰。
    """
    if isinstance(node, list):
        for item in node:
            _harvest_sources(item, sources, seen)
        return
    if not isinstance(node, dict):
        return

    raw_url = node.get("url")
    if isinstance(raw_url, str):
        url = raw_url.strip()
        if url.lower().startswith(("http://", "https://")) and url not in seen:
            seen.add(url)
            title = node.get("title")
            date = node.get("page_age") or node.get("date")
            sources.append(
                {
                    "title": " ".join(str(title).split()) if title else url,
                    "url": url,
                    "date": str(date) if date else None,
                }
            )
    for value in node.values():
        if isinstance(value, (dict, list)):
            _harvest_sources(value, sources, seen)


def _search_count_of(payload: dict, content: list) -> int:
    """本次真的检索了几次。优先用 usage 报的数字，拿不到就数 server_tool_use 块。"""
    usage = payload.get("usage")
    if isinstance(usage, dict):
        server = usage.get("server_tool_use")
        if isinstance(server, dict):
            requests = server.get("web_search_requests")
            if isinstance(requests, bool):
                pass
            elif isinstance(requests, int):
                return max(0, requests)
    return sum(
        1
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "server_tool_use"
        and block.get("name") == WEB_SEARCH_TOOL_NAME
    )


def parse_search_response(payload: dict, purpose: str = "search") -> dict:
    """解析 Anthropic /v1/messages 的返回 -> {"reply", "sources", "search_count"}。

    content 是一个块列表，可能出现四种 type：
        thinking               —— 思考过程，**不进回复**
        text                   —— 真正要给用户看的文字
        server_tool_use        —— 模型发起了一次检索（服务端执行）
        web_search_tool_result —— 检索结果，真实链接就在这里
    未知 type 一律忽略：多一种块不该让整轮请求失败。

    stop_reason == "max_tokens" 只记警告不报错——半截答案仍然比 502 有用，
    但调用方（和日志）必须知道它是半截的。
    """
    if not isinstance(payload, dict):
        raise DeepSeekError(f"检索接口返回结构异常：顶层不是对象（purpose={purpose}）")
    content = payload.get("content")
    if not isinstance(content, list):
        raise DeepSeekError(f"检索接口返回结构异常：缺少 content 列表（purpose={purpose}）")

    texts: list[str] = []
    sources: list[dict] = []
    seen: set[str] = set()
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        elif btype == "web_search_tool_result":
            _harvest_sources(block, sources, seen)

    if payload.get("stop_reason") == "max_tokens":
        logger.warning(
            "检索回复被 max_tokens 截断（purpose=%s）：答案可能不完整，考虑调大 max_tokens",
            purpose,
        )
    return {
        "reply": "\n\n".join(texts).strip(),
        "sources": sources,
        "search_count": _search_count_of(payload, content),
    }


# ---------- 函数调用（tools）辅助 ----------

# 轮数用尽时追加的指令：用手上已有的工具结果作答，不许再编
_FINAL_ANSWER_INSTRUCTION = (
    "You have used all the tool calls available for this turn. Answer the user now, "
    "using only the tool results already shown above and the assessment context. "
    "Cite only URLs that appear in those tool results. If they do not contain what the "
    "user asked for, say so plainly instead of guessing."
)


def _tool_call_name(call: dict) -> str:
    function = call.get("function") if isinstance(call, dict) else None
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return ""


def _tool_call_args(call: dict) -> dict:
    """解析 tool_call 的参数字符串；模型给了非法 JSON 也不能把整轮请求炸掉。"""
    function = call.get("function") if isinstance(call, dict) else None
    raw = function.get("arguments") if isinstance(function, dict) else None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        logger.warning("工具调用参数不是合法 JSON，按空参数处理：%r", raw)
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _run_tool(
    tool_executor: Callable[[str, dict], dict], name: str, args: dict
) -> dict:
    """在线程里执行注入的工具函数；失败转成结构化错误，喂回给模型。"""

    def _call() -> dict:
        return tool_executor(name, args)

    try:
        result = await anyio.to_thread.run_sync(_call)
    except Exception as exc:  # 工具自身崩溃不该让整轮对话 502
        logger.warning("工具 %s 执行失败：%s", name, exc, exc_info=True)
        return {"error": f"tool '{name}' failed: {exc}", "lookup_failed": True}
    return result if isinstance(result, dict) else {"result": result}


class DeepSeekClient:
    """DeepSeek 聊天补全客户端：chat_json 走 JSON 模式，chat_with_tools 走纯文本 + 工具。"""

    async def _request_message(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
        purpose: str,
        json_mode: bool,
        temperature: float,
        tools: list[dict] | None = None,
    ) -> dict:
        """发一次请求并取出整个 choices[0].message。

        取整条 message 而不是只取 content：函数调用时 content 为 null，真正的
        载荷在 tool_calls 里。网络/结构异常统一转成 DeepSeekError。
        """
        settings = get_settings()
        url = settings.deepseek_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": settings.deepseek_model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DeepSeekError(
                f"DeepSeek 接口返回错误状态码 {exc.response.status_code}（purpose={purpose}）"
            ) from exc
        except httpx.HTTPError as exc:
            raise DeepSeekError(
                f"无法连接 DeepSeek 服务（purpose={purpose}）：{exc}"
            ) from exc

        try:
            message = resp.json()["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekError(
                f"DeepSeek 返回结构异常，缺少 choices/message（purpose={purpose}）"
            ) from exc
        if not isinstance(message, dict):
            raise DeepSeekError(f"DeepSeek 返回的 message 不是对象（purpose={purpose}）")
        return message

    async def _request(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
        purpose: str,
        json_mode: bool,
        temperature: float,
    ) -> str:
        """发一次请求并取出 message.content（不带工具的普通调用）。"""
        message = await self._request_message(
            client, messages, purpose, json_mode, temperature
        )
        content = message.get("content")
        if content is None:
            raise DeepSeekError(
                f"DeepSeek 返回结构异常，缺少 choices/message/content（purpose={purpose}）"
            )
        return content

    async def chat_json(
        self,
        system: str,
        user: str,
        purpose: str,
        language: str = "zh-CN",
        tier: str = _DEFAULT_TIER,
    ) -> dict:
        """调用 DeepSeek 并返回解析后的 JSON 对象。

        purpose: "features"（备注症状抽取）或 "advice"（建议生成），
        仅用于日志与 MOCK 数据选择。
        language: 输出语言代码，仅在 MOCK 模式下用于选择对应语言的 advice 假数据；
        真实调用的语言要求由 system prompt 控制。
        tier: 总体风险档位，同样只影响 MOCK 假数据（让演示能区分高低风险）。
        """
        settings = get_settings()

        # MOCK 模式：不发请求，直接返回假数据
        if settings.mock_mode:
            logger.info(
                "MOCK_MODE 开启，返回 %s 假数据（language=%s, tier=%s）",
                purpose,
                language,
                tier,
            )
            if purpose == "features":
                return copy.deepcopy(_MOCK_FEATURES)
            return build_mock_advice(language, tier)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        async with httpx.AsyncClient(timeout=settings.deepseek_timeout) as client:
            for attempt in range(1 + _JSON_RETRIES):
                content = await self._request(
                    client, messages, purpose, json_mode=True, temperature=0.2
                )
                try:
                    data = json.loads(content)
                    if not isinstance(data, dict):
                        raise json.JSONDecodeError("顶层不是 JSON 对象", content, 0)
                    return data
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "DeepSeek 第 %d 次输出无法解析为 JSON（purpose=%s）：%s",
                        attempt + 1,
                        purpose,
                        exc,
                    )
                    # 把解析错误反馈给模型，要求重新输出
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"你上一条回复不是合法的 JSON，解析错误：{exc}。"
                                "请重新只输出一个合法的 JSON 对象，不要包含其他任何内容。"
                            ),
                        }
                    )

        raise DeepSeekError(
            f"DeepSeek 连续 {1 + _JSON_RETRIES} 次未能返回合法 JSON（purpose={purpose}）"
        )


    async def chat_anthropic_search(
        self,
        system: str,
        messages: list[dict],
        language: str = "zh-CN",
        max_uses: int = 2,
        max_tokens: int = SEARCH_MAX_TOKENS,
        purpose: str = "search",
        mock_location: str = "",
    ) -> dict:
        """带联网检索的一次问答，走 **Anthropic 兼容端点**。

        返回 {"reply": str, "sources": [{"title","url","date"|None}], "search_count": int}。

        为什么不能复用 chat_with_tools：DeepSeek 的 web_search 是**服务端工具**，
        只在 /anthropic/v1/messages 上提供。OpenAI 端点没有它，硬塞 tools 也只会
        得到一个普通的函数调用请求。协议不同、鉴权头不同、响应结构不同，
        因此这里是一条独立的路径，而不是给老方法加个参数。

        max_uses <= 0 表示**不挂检索工具**（此时就是一次普通的 Anthropic 调用）——
        调用方据此实现「没有地点就不检索」，不必在这里判断产品规则。

        mock_location 仅在 MOCK_MODE 下使用：决定演示数据里那份来源清单要不要给。
        """
        settings = get_settings()

        if settings.mock_mode:
            logger.info(
                "MOCK_MODE 开启，返回检索假数据（language=%s, location=%r）",
                language,
                mock_location,
            )
            return build_mock_search(language, mock_location)

        url = settings.deepseek_base_url.rstrip("/") + ANTHROPIC_MESSAGES_PATH
        headers = {
            "x-api-key": settings.deepseek_api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": settings.deepseek_model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": to_anthropic_messages(messages),
        }
        if max_uses > 0:
            payload["tools"] = [
                {
                    "type": WEB_SEARCH_TOOL_TYPE,
                    "name": WEB_SEARCH_TOOL_NAME,
                    "max_uses": max_uses,
                }
            ]

        try:
            async with httpx.AsyncClient(timeout=settings.deepseek_timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise DeepSeekError(
                f"检索接口返回错误状态码 {exc.response.status_code}（purpose={purpose}）"
            ) from exc
        except httpx.HTTPError as exc:
            raise DeepSeekError(f"无法连接检索接口（purpose={purpose}）：{exc}") from exc
        except ValueError as exc:  # 响应不是 JSON
            raise DeepSeekError(f"检索接口返回的不是 JSON（purpose={purpose}）") from exc

        outcome = parse_search_response(data, purpose=purpose)
        logger.info(
            "检索完成（purpose=%s）：检索 %d 次，取回 %d 条来源，回复 %d 字",
            purpose,
            outcome["search_count"],
            len(outcome["sources"]),
            len(outcome["reply"]),
        )
        return outcome

    async def chat_with_tools(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_executor: Callable[[str, dict], dict],
        language: str = "zh-CN",
        tier: str = "low",
        max_rounds: int = _DEFAULT_TOOL_ROUNDS,
        purpose: str = "chat",
        mock_probe: str = "",
    ) -> dict:
        """带函数调用的对话。返回 {"reply": str, "tool_results": [...]}。

        tool_executor(name, args) -> dict 由流水线注入：客户端只负责搬运
        tool_calls 与 tool 消息，不知道也不关心工具到底查了什么。工具是同步函数，
        放到线程里跑，避免一次 8 秒的外部请求把事件循环钉住。

        max_rounds 是**模型调用**的轮数上限。轮数用尽时不再给它 tools，而是把
        已经拿到的工具结果留在上下文里、追加一句「现在就回答」——宁可拿一个
        基于已有数据的答案，也不要无限循环下去。

        tool_results 里每项是 {"name", "arguments", "result"}，流水线据此算出
        本轮允许引用的链接白名单（见 verifier.verify_chat_reply）。

        mock_probe 只在 MOCK 模式下使用：本轮问题与最近历史的原文，用来判断
        该不该模拟一次工具调用。真实模式完全忽略它——是否调用工具由模型决定。
        """
        settings = get_settings()

        if settings.mock_mode:
            return await self._mock_tool_conversation(
                mock_probe, tool_executor, language, tier
            )

        convo: list[dict] = [{"role": "system", "content": system}]
        convo += [dict(m) for m in messages]
        tool_results: list[dict] = []

        async with httpx.AsyncClient(timeout=settings.deepseek_timeout) as client:
            for round_index in range(max(1, max_rounds)):
                message = await self._request_message(
                    client, convo, purpose, json_mode=False, temperature=0.4, tools=tools
                )
                calls = message.get("tool_calls") or []
                if not calls:
                    return {
                        "reply": (message.get("content") or "").strip(),
                        "tool_results": tool_results,
                    }

                logger.info(
                    "第 %d 轮：模型请求调用 %d 个工具（%s）",
                    round_index + 1,
                    len(calls),
                    ", ".join(_tool_call_name(c) for c in calls),
                )
                convo.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or "",
                        "tool_calls": calls,
                    }
                )
                for call in calls:
                    name = _tool_call_name(call)
                    args = _tool_call_args(call)
                    result = await _run_tool(tool_executor, name, args)
                    tool_results.append(
                        {"name": name, "arguments": args, "result": result}
                    )
                    convo.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(call.get("id") or ""),
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

            # 轮数用尽：把已有结果摆上桌，要求立刻作答（这一次不再给 tools）
            convo.append({"role": "user", "content": _FINAL_ANSWER_INSTRUCTION})
            content = await self._request(
                client, convo, purpose, json_mode=False, temperature=0.4
            )

        return {"reply": (content or "").strip(), "tool_results": tool_results}

    async def _mock_tool_conversation(
        self,
        probe: str,
        tool_executor: Callable[[str, dict], dict],
        language: str,
        tier: str,
    ) -> dict:
        """MOCK 模式：问题里提到已知地名就模拟一轮工具调用，否则走原来的假回复。

        关键是**真的调用一次 tool_executor**：演示回复里的链接因此确实来自
        工具结果，而不是写死在模板里——这样 MOCK 也能验证「不许编造链接」这条
        不变量，而不是绕过它。
        """
        location = find_location(probe or "")
        if location is None:
            logger.info("MOCK_MODE 开启，问题未提到已知地区，返回通用假回复")
            return {"reply": build_mock_chat_reply(language, tier), "tool_results": []}

        logger.info("MOCK_MODE 开启，模拟一轮工具调用：location=%s", location)
        args = {"location": location}
        result = await _run_tool(tool_executor, INTEL_TOOL_NAME, args)
        return {
            "reply": build_mock_tool_reply(language, result),
            "tool_results": [
                {"name": INTEL_TOOL_NAME, "arguments": args, "result": result}
            ],
        }
