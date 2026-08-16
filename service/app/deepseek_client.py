"""DeepSeek 客户端：OpenAI 兼容的 /chat/completions 调用。

两种调用方式：
  chat_json —— 强制 response_format=json_object，解析失败会把错误回喂给模型重试；
               用于「备注症状抽取」与「建议生成」。
  chat_text —— 普通文本输出，不加 response_format；用于 /api/chat 的追问对话
               （聊天回复应当是散文，不是 JSON）。

MOCK_MODE=true 时不发任何网络请求，直接返回可信假数据，便于本地演示与测试。
假建议按**风险档位**（low/medium/high）分三套，让演示能看出差异。
"""

import copy
import json
import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# JSON 解析失败时的最大重试次数（不含首次请求）
_JSON_RETRIES = 2


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


def build_mock_advice(language: str, tier: str) -> dict:
    """按语言与风险档位组装假建议（键顺序与 Advice 模型一致）。"""
    lang, level = _normalise(language, tier)
    return {
        "summary": _MOCK_SUMMARY[lang][level],
        "advice": {
            "medical": list(_MOCK_MEDICAL[lang][level]),
            "monitoring": list(_MOCK_MONITORING[lang]),
            "protection": list(_MOCK_PROTECTION[lang]),
        },
    }


def build_mock_chat_reply(language: str, tier: str) -> str:
    """按语言与风险档位组装假聊天回复（引用用户自己的风险等级）。"""
    lang, level = _normalise(language, tier)
    return _MOCK_CHAT_TEMPLATES[lang].format(tier=_MOCK_CHAT_TIER_LABELS[lang][level])


class DeepSeekClient:
    """DeepSeek 聊天补全客户端：chat_json 走 JSON 模式，chat_text 走纯文本。"""

    async def _request(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
        purpose: str,
        json_mode: bool,
        temperature: float,
    ) -> str:
        """发一次请求并取出 message.content；网络/结构异常统一转成 DeepSeekError。"""
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
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekError(
                f"DeepSeek 返回结构异常，缺少 choices/message/content（purpose={purpose}）"
            ) from exc

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

    async def chat_text(
        self,
        system: str,
        user: str,
        purpose: str = "chat",
        language: str = "zh-CN",
        tier: str = "low",
    ) -> str:
        """调用 DeepSeek 并返回纯文本回复（不加 response_format）。

        用于 /api/chat：聊天回复应当是给人读的散文，套一层 JSON 只会让模型
        把精力花在格式上，也让空白与换行更难保留。没有 JSON 就没有解析失败，
        因此这里不重试。
        """
        settings = get_settings()

        if settings.mock_mode:
            logger.info(
                "MOCK_MODE 开启，返回 %s 假回复（language=%s, tier=%s）",
                purpose,
                language,
                tier,
            )
            return build_mock_chat_reply(language, tier)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        async with httpx.AsyncClient(timeout=settings.deepseek_timeout) as client:
            content = await self._request(
                client, messages, purpose, json_mode=False, temperature=0.4
            )

        text = (content or "").strip()
        if not text:
            raise DeepSeekError(f"DeepSeek 返回了空回复（purpose={purpose}）")
        return text
