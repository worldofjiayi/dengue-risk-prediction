"""评估流水线：问卷 -> 特征编码 -> 三模型打分 -> DeepSeek 生成建议 -> 结果组装。

特征编码是**确定性的**（app.ml_model.encode_features），这是权威来源。
DeepSeek 的第一次调用只做一件补充工作：从用户的自由文本备注里识别出
「描述了但没勾选」的症状，把对应项从 unknown 提升为 yes。
该步骤失败不影响评估——直接沿用确定性编码结果。

除模型评分外，本模块还产出两条**与模型无关的规则判断**：
  warning_signs    —— WHO 登革热警示征象（VOMITO / PETEQUIA_N）
  exposure_context —— 流行病学暴露背景（身边有确诊病例 / 去过暴发地区 / 周围发热聚集）
两者都不进入 26 维特征向量，也不参与打分，只与评分并列呈现。

本模块另外承载 /api/chat 的追问对话（run_chat）——无状态，上下文由前端回传。

日志中不记录 notes 原文，避免用户敏感信息进入日志。
"""

import logging
import time

from app.config import get_settings
from app.deepseek_client import DeepSeekClient, DeepSeekError
from app.eval_log import log_assessment
from app.ml_model import encode_features, get_epi_week, get_model
from app.prompt_builder import (
    build_advice_prompt,
    build_chat_prompt,
    build_feature_prompt,
)
from app.schemas import (
    DISCLAIMERS,
    EXPOSURE_CODES,
    HIGH_EXPOSURE_CODES,
    MEDIUM_EXPOSURE_CODES,
    MODEL_NOTES,
    SYMPTOM_CODES,
    WARNING_SIGN_CODES,
    Advice,
    AssessmentResult,
    ChatRequest,
    ChatResponse,
    ExposureContext,
    FormInput,
)

logger = logging.getLogger(__name__)

# 风险等级由低到高，用于取「三个模型中最高的那一档」
_LEVEL_ORDER: tuple[str, ...] = ("low", "medium", "high")

# summary 缺失时的兜底文案（五语言）
_FALLBACK_SUMMARY = {
    "zh-CN": "已完成风险评估，请结合下方建议做好防蚊防护与健康监测。",
    "zh-TW": "已完成風險評估，請結合下方建議做好防蚊防護與健康監測。",
    "en": "Assessment complete. Please follow the guidance below on mosquito protection and monitoring.",
    "es": "Evaluación completada. Siga las recomendaciones sobre protección contra mosquitos y vigilancia.",
    "pt": "Avaliação concluída. Siga as orientações abaixo sobre proteção contra mosquitos e monitoramento.",
}


# /api/chat 回复为空时的兜底文案（五语言）
_FALLBACK_REPLY = {
    "zh-CN": "抱歉，我这次没能生成有效回复。请换个说法再问一次；若症状加重请及时就医。",
    "zh-TW": "抱歉，這次未能產生有效回覆。請換個說法再問一次；若症狀加重請及時就醫。",
    "en": "Sorry, I could not produce a useful answer this time. Please rephrase and ask again; seek medical care if your symptoms worsen.",
    "es": "Lo siento, no pude generar una respuesta útil. Reformule la pregunta e inténtelo de nuevo; si sus síntomas empeoran, busque atención médica.",
    "pt": "Desculpe, não consegui gerar uma resposta útil. Reformule a pergunta e tente novamente; se os sintomas piorarem, procure atendimento médico.",
}


def overall_tier(levels: list[str]) -> str:
    """取一组风险等级中最高的一档（high > medium > low）。

    用于给建议生成与追问对话一个「总体档位」：只要有任何一个模型报到 high，
    整体口径就按 high 走——宁可多提醒一次，也不要让高分被两个低分平均掉。
    """
    best = "low"
    for level in levels:
        if level in _LEVEL_ORDER and _LEVEL_ORDER.index(level) > _LEVEL_ORDER.index(best):
            best = level
    return best


def evaluate_exposure(form: FormInput) -> ExposureContext:
    """流行病学暴露的**规则判断**（不经过任何模型）。

        high   —— CONFIRMED_CASE 或 OUTBREAK_TRAVEL 为 yes
        medium —— FEVER_CLUSTER 为 yes 且未达 high
        low    —— 其余情况

    只有明确回答 yes 才算数：与症状编码一样，「不知道」不等于「有」，
    不能凭用户的不确定去抬高风险提示。

    为什么不并入模型：SINAN 通报数据里没有这三个变量，逻辑回归没有对应系数，
    任何权重都只能是编出来的数字。分开呈现，用户和医生都能看清哪部分来自
    数据拟合、哪部分来自流行病学常识。
    """
    answers = form.exposure
    factors = [c for c in EXPOSURE_CODES if answers.get(c) == "yes"]
    if any(answers.get(c) == "yes" for c in HIGH_EXPOSURE_CODES):
        level = "high"
    elif any(answers.get(c) == "yes" for c in MEDIUM_EXPOSURE_CODES):
        level = "medium"
    else:
        level = "low"
    return ExposureContext(level=level, factors=factors)


async def _infer_notes_symptoms(
    form: FormInput, client: DeepSeekClient
) -> FormInput:
    """用 DeepSeek 从备注里补充症状；任何失败都返回原表单。"""
    system, user = build_feature_prompt(form)
    try:
        raw = await client.chat_json(system, user, purpose="features")
    except DeepSeekError:
        logger.warning("备注症状抽取调用失败，沿用确定性编码", exc_info=True)
        return form

    infer = raw.get("infer")
    if not isinstance(infer, dict) or not infer:
        return form

    updated = dict(form.symptoms)
    applied: list[str] = []
    for code, value in infer.items():
        # 只允许把「尚未作答」提升为「有」，绝不推翻用户明确的回答
        if code in SYMPTOM_CODES and value == "yes" and updated.get(code) == "unknown":
            updated[code] = "yes"
            applied.append(code)

    if not applied:
        return form
    logger.info("备注中识别到额外症状：%s", applied)
    return form.model_copy(update={"symptoms": updated})


async def run_assessment(form: FormInput) -> AssessmentResult:
    """完整评估流程，任何一步失败都会抛出异常，由上层路由转成 HTTP 错误。"""
    settings = get_settings()
    client = DeepSeekClient()
    t0 = time.perf_counter()

    # ---- 步骤 1：确定性特征编码 ----
    # 有备注且非 MOCK 时，先让 DeepSeek 从备注里补充症状
    if form.notes.strip() and not settings.mock_mode:
        form = await _infer_notes_symptoms(form, client)
    features = encode_features(form)
    epi_week = get_epi_week()
    logger.info(
        "步骤1 特征编码完成，耗时 %.2fs，epi_week=%d",
        time.perf_counter() - t0,
        epi_week,
    )

    # WHO 警示征象：规则判断，独立于模型评分（见 schemas.WARNING_SIGN_CODES）
    warning_signs = [c for c in WARNING_SIGN_CODES if form.symptoms.get(c) == "yes"]
    if warning_signs:
        logger.info("用户报告 WHO 警示征象：%s", warning_signs)

    # 流行病学暴露：同样是规则判断，不进入特征向量（见 evaluate_exposure）
    exposure = evaluate_exposure(form)
    if exposure.factors:
        logger.info("流行病学暴露：level=%s factors=%s", exposure.level, exposure.factors)

    # ---- 步骤 2：三个模型打分 ----
    t1 = time.perf_counter()
    model = get_model()
    scores = model.score_all(features)
    explanations = model.explain_all(features)
    tier = overall_tier([s.level for s in scores.values()])
    logger.info(
        "步骤2 模型打分完成，耗时 %.2fs，A=%.1f(%s) B=%.1f(%s) B2=%.1f(%s)，总体档位=%s",
        time.perf_counter() - t1,
        scores["A"].score, scores["A"].level,
        scores["B"].score, scores["B"].level,
        scores["B2"].score, scores["B2"].level,
        tier,
    )

    # ---- 步骤 3：DeepSeek 生成建议 ----
    t2 = time.perf_counter()
    adv_system, adv_user = build_advice_prompt(
        form, scores, epi_week, warning_signs, exposure
    )
    raw_advice = await client.chat_json(
        adv_system, adv_user, purpose="advice", language=form.language, tier=tier
    )
    try:
        advice = Advice.model_validate(raw_advice.get("advice", {}))
    except Exception as exc:  # pydantic ValidationError 及结构异常
        raise DeepSeekError("DeepSeek 建议生成结果不符合要求，无法完成评估") from exc
    summary = str(raw_advice.get("summary", "")).strip()
    if not summary:
        summary = _FALLBACK_SUMMARY[form.language]
    logger.info("步骤3 建议生成完成，耗时 %.2fs", time.perf_counter() - t2)

    # ---- 步骤 4：组装结果 ----
    result = AssessmentResult(
        dengue=scores["A"],
        worsening=scores["B"],
        severe=scores["B2"],
        epi_week=epi_week,
        warning_signs=warning_signs,
        exposure_context=exposure,
        summary=summary,
        advice=advice,
        explanations=explanations,
        disclaimer=DISCLAIMERS[form.language],
        model_note=MODEL_NOTES[form.language],
    )
    logger.info("评估完成，总耗时 %.2fs", time.perf_counter() - t0)

    log_assessment(form, features, scores, epi_week, exposure)
    return result


async def run_chat(req: ChatRequest) -> ChatResponse:
    """追问对话：无状态，上下文与历史全部由前端回传。

    失败（DeepSeekError）向上抛出，由路由转成 502。
    """
    t0 = time.perf_counter()
    tier = overall_tier(
        [
            block.level
            for block in (req.context.dengue, req.context.worsening, req.context.severe)
            if block is not None
        ]
    )
    system, user = build_chat_prompt(req)
    reply = await DeepSeekClient().chat_text(
        system, user, purpose="chat", language=req.language, tier=tier
    )
    reply = reply.strip() or _FALLBACK_REPLY[req.language]
    logger.info(
        "追问对话完成，耗时 %.2fs（language=%s, tier=%s, history=%d 条）",
        time.perf_counter() - t0,
        req.language,
        tier,
        len(req.history),
    )
    return ChatResponse(reply=reply)
