"""评估流水线：问卷 -> 特征编码 -> 三模型打分 -> DeepSeek 生成建议 -> 结果组装。

特征编码是**确定性的**（app.ml_model.encode_features），这是权威来源。
DeepSeek 的第一次调用只做一件补充工作：从用户的自由文本备注里识别出
「描述了但没勾选」的症状，把对应项从 unknown 提升为 yes。
该步骤失败不影响评估——直接沿用确定性编码结果。

日志中不记录 notes 原文，避免用户敏感信息进入日志。
"""

import logging
import time

from app.config import get_settings
from app.deepseek_client import DeepSeekClient, DeepSeekError
from app.eval_log import log_assessment
from app.ml_model import encode_features, get_epi_week, get_model
from app.prompt_builder import build_advice_prompt, build_feature_prompt
from app.schemas import (
    DISCLAIMERS,
    MODEL_NOTES,
    SYMPTOM_CODES,
    WARNING_SIGN_CODES,
    Advice,
    AssessmentResult,
    FormInput,
)

logger = logging.getLogger(__name__)

# summary 缺失时的兜底文案（五语言）
_FALLBACK_SUMMARY = {
    "zh-CN": "已完成风险评估，请结合下方建议做好防蚊防护与健康监测。",
    "zh-TW": "已完成風險評估，請結合下方建議做好防蚊防護與健康監測。",
    "en": "Assessment complete. Please follow the guidance below on mosquito protection and monitoring.",
    "es": "Evaluación completada. Siga las recomendaciones sobre protección contra mosquitos y vigilancia.",
    "pt": "Avaliação concluída. Siga as orientações abaixo sobre proteção contra mosquitos e monitoramento.",
}


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

    # ---- 步骤 2：三个模型打分 ----
    t1 = time.perf_counter()
    scores = get_model().score_all(features)
    logger.info(
        "步骤2 模型打分完成，耗时 %.2fs，A=%.1f(%s) B=%.1f(%s) B2=%.1f(%s)",
        time.perf_counter() - t1,
        scores["A"].score, scores["A"].level,
        scores["B"].score, scores["B"].level,
        scores["B2"].score, scores["B2"].level,
    )

    # ---- 步骤 3：DeepSeek 生成建议 ----
    t2 = time.perf_counter()
    adv_system, adv_user = build_advice_prompt(form, scores, epi_week, warning_signs)
    raw_advice = await client.chat_json(
        adv_system, adv_user, purpose="advice", language=form.language
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
        summary=summary,
        advice=advice,
        disclaimer=DISCLAIMERS[form.language],
        model_note=MODEL_NOTES[form.language],
    )
    logger.info("评估完成，总耗时 %.2fs", time.perf_counter() - t0)

    log_assessment(form, features, scores, epi_week)
    return result
