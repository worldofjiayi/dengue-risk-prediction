"""DeepSeek 客户端：OpenAI 兼容的 /chat/completions 调用，强制 JSON 输出。

MOCK_MODE=true 时不发任何网络请求，直接返回可信假数据，便于本地演示与测试。
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

# 各语言的 advice 假数据（登革热专属），键为 FormInput.language 的语言代码
_MOCK_ADVICE = {
    "zh-CN": {
        "summary": "根据您填写的症状与病史，当前登革热相关风险处于中等水平，建议加强防蚊并密切观察病情变化。",
        "advice": {
            "protection": [
                "清除住所周边的积水容器（花盆托盘、水桶、废旧轮胎），从源头减少蚊虫孳生。",
                "使用纱窗、蚊帐，白天也要防蚊——传播登革热的伊蚊主要在日间叮咬。",
                "外出时穿浅色长袖衣裤，在裸露皮肤涂抹驱蚊剂。",
            ],
            "medical": [
                "如出现持续呕吐、剧烈腹痛、牙龈或鼻腔出血、嗜睡或烦躁不安，请立即就医。",
                "退热镇痛请遵医嘱，避免自行服用阿司匹林或布洛芬类药物，以免增加出血风险。",
            ],
            "monitoring": [
                "每日早晚测量体温并记录，注意退热后 24-48 小时是病情变化的关键窗口。",
                "保证充足饮水，留意尿量是否明显减少。",
                "观察皮肤有无新出现的出血点或瘀斑。",
            ],
        },
    },
    "zh-TW": {
        "summary": "根據您填寫的症狀與病史，目前登革熱相關風險處於中等水準，建議加強防蚊並密切觀察病情變化。",
        "advice": {
            "protection": [
                "清除住所周邊的積水容器（花盆底盤、水桶、廢輪胎），從源頭減少病媒蚊孳生。",
                "使用紗窗、蚊帳，白天也要防蚊——傳播登革熱的斑蚊主要在日間叮咬。",
                "外出時穿淺色長袖衣褲，在裸露皮膚塗抹防蚊液。",
            ],
            "medical": [
                "如出現持續嘔吐、劇烈腹痛、牙齦或鼻腔出血、嗜睡或躁動不安，請立即就醫。",
                "退燒止痛請遵醫囑，避免自行服用阿斯匹靈或布洛芬類藥物，以免增加出血風險。",
            ],
            "monitoring": [
                "每日早晚量測體溫並記錄，注意退燒後 24-48 小時是病情變化的關鍵時期。",
                "確保充足飲水，留意尿量是否明顯減少。",
                "觀察皮膚有無新出現的出血點或瘀斑。",
            ],
        },
    },
    "en": {
        "summary": (
            "Based on the symptoms and history you provided, your dengue-related risk appears "
            "moderate. Strengthen mosquito protection and monitor your condition closely."
        ),
        "advice": {
            "protection": [
                "Remove standing water around your home (plant saucers, buckets, old tyres) to stop mosquitoes breeding.",
                "Use window screens and bed nets, and protect yourself during the day — the Aedes mosquitoes that spread dengue bite mainly in daylight.",
                "Wear light-coloured long sleeves and trousers outdoors, and apply repellent to exposed skin.",
            ],
            "medical": [
                "Seek care immediately if you develop persistent vomiting, severe abdominal pain, bleeding gums or nose, drowsiness, or restlessness.",
                "Take fever or pain medication only as advised by a clinician, and avoid self-medicating with aspirin or ibuprofen, which can increase bleeding risk.",
            ],
            "monitoring": [
                "Record your temperature every morning and evening; the 24–48 hours after fever subsides is the critical window.",
                "Keep well hydrated and watch for a marked drop in urine output.",
                "Check your skin for new bleeding spots or bruising.",
            ],
        },
    },
    "es": {
        "summary": (
            "Según los síntomas y antecedentes que indicó, su riesgo relacionado con el dengue "
            "es moderado. Refuerce la protección contra mosquitos y vigile de cerca su evolución."
        ),
        "advice": {
            "protection": [
                "Elimine los recipientes con agua estancada alrededor de su vivienda (platos de macetas, baldes, neumáticos) para cortar la cría de mosquitos.",
                "Use mosquiteros en ventanas y camas, y protéjase también de día: el mosquito Aedes que transmite el dengue pica principalmente durante el día.",
                "Al salir, use ropa clara de manga larga y aplique repelente en la piel expuesta.",
            ],
            "medical": [
                "Acuda de inmediato a un centro de salud si presenta vómitos persistentes, dolor abdominal intenso, sangrado de encías o nariz, somnolencia o inquietud.",
                "Tome analgésicos o antipiréticos solo bajo indicación médica y evite automedicarse con aspirina o ibuprofeno, que pueden aumentar el riesgo de sangrado.",
            ],
            "monitoring": [
                "Mida y registre su temperatura por la mañana y por la noche; las 24–48 horas posteriores a la caída de la fiebre son el periodo crítico.",
                "Manténgase bien hidratado y observe si disminuye notablemente la cantidad de orina.",
                "Revise su piel en busca de nuevos puntos de sangrado o moretones.",
            ],
        },
    },
    "pt": {
        "summary": (
            "Com base nos sintomas e antecedentes informados, seu risco relacionado à dengue "
            "está em nível moderado. Reforce a proteção contra mosquitos e acompanhe de perto sua evolução."
        ),
        "advice": {
            "protection": [
                "Elimine recipientes com água parada ao redor de casa (pratos de vasos, baldes, pneus velhos) para impedir a criação do mosquito.",
                "Use telas nas janelas e mosquiteiro, e proteja-se também durante o dia — o Aedes, transmissor da dengue, pica principalmente à luz do dia.",
                "Ao sair, use roupas claras de manga comprida e aplique repelente na pele exposta.",
            ],
            "medical": [
                "Procure atendimento imediatamente se surgirem vômitos persistentes, dor abdominal intensa, sangramento de gengiva ou nariz, sonolência ou agitação.",
                "Use antitérmicos ou analgésicos apenas com orientação médica e evite automedicação com aspirina ou ibuprofeno, que podem aumentar o risco de sangramento.",
            ],
            "monitoring": [
                "Meça e registre a temperatura de manhã e à noite; as 24–48 horas após a queda da febre são o período crítico.",
                "Mantenha-se bem hidratado e observe se o volume de urina diminui de forma acentuada.",
                "Verifique a pele em busca de novos pontos de sangramento ou manchas roxas.",
            ],
        },
    },
}


class DeepSeekClient:
    """DeepSeek 聊天补全客户端，只支持 JSON 输出场景。"""

    async def chat_json(
        self, system: str, user: str, purpose: str, language: str = "zh-CN"
    ) -> dict:
        """调用 DeepSeek 并返回解析后的 JSON 对象。

        purpose: "features"（备注症状抽取）或 "advice"（建议生成），
        仅用于日志与 MOCK 数据选择。
        language: 输出语言代码，仅在 MOCK 模式下用于选择对应语言的 advice 假数据；
        真实调用的语言要求由 system prompt 控制。
        """
        settings = get_settings()

        # MOCK 模式：不发请求，直接返回假数据
        if settings.mock_mode:
            logger.info("MOCK_MODE 开启，返回 %s 假数据（language=%s）", purpose, language)
            if purpose == "features":
                return copy.deepcopy(_MOCK_FEATURES)
            return copy.deepcopy(_MOCK_ADVICE.get(language, _MOCK_ADVICE["zh-CN"]))

        url = settings.deepseek_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        async with httpx.AsyncClient(timeout=settings.deepseek_timeout) as client:
            for attempt in range(1 + _JSON_RETRIES):
                payload = {
                    "model": settings.deepseek_model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                }
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
                    content = resp.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise DeepSeekError(
                        f"DeepSeek 返回结构异常，缺少 choices/message/content（purpose={purpose}）"
                    ) from exc

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
