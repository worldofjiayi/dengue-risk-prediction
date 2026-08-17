"""评估流水线：问卷 -> 特征编码 -> 三模型打分 -> DeepSeek 生成建议 -> 结果组装。

特征编码是**确定性的**（app.ml_model.encode_features），这是权威来源。
DeepSeek 的第一次调用只做一件补充工作：从用户的自由文本备注里识别出
「描述了但没勾选」的症状，把对应项从 unknown 提升为 yes。
该步骤失败不影响评估——直接沿用确定性编码结果。

除模型评分外，本模块还产出两条**与模型无关的规则判断**：
  warning_signs    —— WHO 登革热警示征象（VOMITO / PETEQUIA_N）
  exposure_context —— 流行病学暴露背景（身边有确诊病例 / 去过暴发地区 / 周围发热聚集）
两者都不进入 26 维特征向量，也不参与打分，只与评分并列呈现。

本模块另外承载 /api/chat 的追问对话（run_chat）——无状态，上下文由前端回传，
并给模型挂上一个可自主调用的工具（app.intel.lookup_dengue_context）。

**生成之后还有一道闸门**：所有 LLM 文本在返回前都要过 app.verifier 的规则校验
（剂量、感染概率、就医紧迫性、语言一致性、结构、编造链接）。不通过就带着违规
说明重问一次；还不通过就退回模板 / 兜底文案。因此：

  - 建议生成失败**不再让整次评估失败**。评分是本地算出来的，是这个服务真正
    值钱的部分；因为一句自然语言拿不到就把 200 变成 502，是拿用户已经得到的
    结果去赔一个可以替代的段落。改为返回 200 + 模板建议 + advice_source=template。
  - /api/chat 没有可退的东西（回复本身就是全部产出），仍然 502。

日志中不记录 notes 原文，避免用户敏感信息进入日志。
"""

import logging
import time

from app.config import get_settings
from app.deepseek_client import DeepSeekClient, DeepSeekError, fallback_advice
from app.eval_log import log_assessment
from app.intel import INTEL_TOOL_NAME, lookup_dengue_context
from app.ml_model import encode_features, get_epi_week, get_model
from app.prompt_builder import (
    build_advice_prompt,
    build_chat_prompt,
    build_chat_tools,
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
    Source,
)
from app.verifier import (
    Violation,
    format_violations,
    verify_advice,
    verify_chat_reply,
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


# 追问回复两次都没通过输出校验时的兜底（五语言）。
#
# 刻意不试图「大致回答一下」：校验没过说明这段文字里有不该出现的东西，
# 把它端出去比不回答更糟，宁可承认这一轮不可靠。
# 「回复为空」现在也是一条违规（verifier 的 empty 规则），因此空回复与其他
# 违规共用这同一条路，不再需要第二份措辞不同、迟早会各自漂移的兜底句。
_UNRELIABLE_REPLY = {
    "zh-CN": "抱歉，我这次无法给出可靠的回答，请咨询当地的医疗机构或公共卫生服务。若症状加重请尽快就医。",
    "zh-TW": "抱歉，這次無法給出可靠的回覆，請諮詢當地醫療院所或公共衛生服務。若症狀加重請儘快就醫。",
    "en": "I can't produce a reliable answer right now — please consult a local health service. If your symptoms worsen, seek medical care promptly.",
    "es": "No puedo dar una respuesta fiable en este momento; consulte a un servicio de salud local. Si sus síntomas empeoran, busque atención médica lo antes posible.",
    "pt": "Não consigo dar uma resposta confiável agora — consulte um serviço de saúde local. Se os sintomas piorarem, procure atendimento médico o quanto antes.",
}

# 建议生成的输出校验最多重问几次（不含首次）
_ADVICE_VERIFY_RETRIES = 1
# 追问回复的输出校验最多重问几次（不含首次）
_CHAT_VERIFY_RETRIES = 1


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


def _parse_advice(raw: dict, language: str) -> tuple[Advice, str]:
    """把模型返回的 JSON 解析成 (Advice, summary)。结构不合法就抛 DeepSeekError。"""
    try:
        advice = Advice.model_validate(raw.get("advice", {}))
    except Exception as exc:  # pydantic ValidationError 及结构异常
        raise DeepSeekError("DeepSeek 建议生成结果不符合要求") from exc
    summary = str(raw.get("summary", "")).strip() or _FALLBACK_SUMMARY[language]
    return advice, summary


def _template_advice(language: str, tier: str) -> tuple[Advice, str]:
    """兜底：与 MOCK 演示共用同一份分档模板（deepseek_client.fallback_advice）。"""
    raw = fallback_advice(language, tier)
    return Advice.model_validate(raw["advice"]), raw["summary"]


def _log_violations(where: str, violations: list[Violation]) -> None:
    logger.warning(
        "%s 未通过输出校验：%s", where, "；".join(v.code for v in violations)
    )
    for violation in violations:
        logger.debug("  %s", violation)


async def _produce_advice(
    form: FormInput,
    scores: dict,
    epi_week: int,
    warning_signs: list[str],
    exposure: ExposureContext,
    tier: str,
    client: DeepSeekClient,
    settings,
) -> tuple[Advice, str, str]:
    """生成建议并保证它通过输出校验。返回 (advice, summary, advice_source)。

    真实模式：生成 -> 校验 -> 带违规说明重问一次 -> 再校验 -> 仍不过就退回模板。
    MOCK 模式：直接用模板，但**照样跑一遍校验**——校验是纯规则、零成本，
    而模板正是真实模式失败时要端给用户的东西，它必须永远是干净的。
    这条路径由 tests 里「5 语言 × 3 档位 × 0 违规」那条测试守着。
    """
    language = form.language
    adv_system, adv_user = build_advice_prompt(
        form, scores, epi_week, warning_signs, exposure
    )

    if settings.mock_mode:
        raw = await client.chat_json(
            adv_system, adv_user, purpose="advice", language=language, tier=tier
        )
        advice, summary = _parse_advice(raw, language)
        violations = verify_advice(advice, summary, language, tier, warning_signs)
        if violations:  # 不该发生：模板文案有问题，要在日志里吼出来
            _log_violations("MOCK 模板建议", violations)
        return advice, summary, "template"

    user_prompt = adv_user
    for attempt in range(1 + _ADVICE_VERIFY_RETRIES):
        try:
            raw = await client.chat_json(
                adv_system, user_prompt, purpose="advice", language=language, tier=tier
            )
            advice, summary = _parse_advice(raw, language)
        except DeepSeekError as exc:
            logger.error("建议生成失败（第 %d 次）：%s", attempt + 1, exc)
            break

        violations = verify_advice(advice, summary, language, tier, warning_signs)
        if not violations:
            return advice, summary, "llm"
        _log_violations(f"建议生成第 {attempt + 1} 次", violations)
        # 把违规说明拼回提示词，让模型只改这些地方
        user_prompt = adv_user + "\n\n" + format_violations(violations, as_json=True)

    logger.warning("建议退回模板文案（language=%s, tier=%s）", language, tier)
    advice, summary = _template_advice(language, tier)
    return advice, summary, "template"


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

    # ---- 步骤 3：DeepSeek 生成建议（生成 -> 校验 -> 重问 -> 兜底） ----
    t2 = time.perf_counter()
    advice, summary, advice_source = await _produce_advice(
        form, scores, epi_week, warning_signs, exposure, tier, client, settings
    )
    logger.info(
        "步骤3 建议生成完成，耗时 %.2fs，来源=%s",
        time.perf_counter() - t2,
        advice_source,
    )

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
        advice_source=advice_source,
    )
    logger.info("评估完成，总耗时 %.2fs", time.perf_counter() - t0)

    log_assessment(form, features, scores, epi_week, exposure)
    return result


def _make_tool_executor(collected: list[dict]):
    """构造注入给客户端的工具执行器。

    客户端只知道「有个可调用的函数」，工具到底查什么、参数怎么清洗、结果往哪存，
    全在这里——传输层与领域逻辑分开，客户端才能被独立测试。
    """

    def execute(name: str, args: dict) -> dict:
        if name != INTEL_TOOL_NAME:
            logger.warning("模型请求了未知工具：%r", name)
            return {"error": f"unknown tool '{name}'", "lookup_failed": True}
        location = str((args or {}).get("location", "")).strip()[:120]
        result = lookup_dengue_context(location)
        collected.append(result)
        return result

    return execute


def _sources_from(tool_results: list[dict]) -> list[Source]:
    """把工具结果里的 WHO 通报收成引用列表（按 url 去重，保持顺序）。"""
    sources: list[Source] = []
    seen: set[str] = set()
    for entry in tool_results:
        result = entry.get("result") if isinstance(entry, dict) else None
        if not isinstance(result, dict):
            continue
        for notice in result.get("who_notices") or []:
            url = str(notice.get("url", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append(
                Source(
                    title=str(notice.get("title", "")),
                    date=str(notice.get("date", "")),
                    url=url,
                )
            )
    return sources


async def run_chat(req: ChatRequest) -> ChatResponse:
    """追问对话：无状态，上下文与历史全部由前端回传。

    模型可以自主调用 lookup_dengue_context 查某地的登革热背景；回复在返回前
    要过 verify_chat_reply，其中 allowed_urls **只包含这一轮工具真正返回过的**
    链接。校验不过就带违规说明重问一次，还不过就换成本地化的兜底句。

    失败（DeepSeekError）向上抛出，由路由转成 502——聊天没有别的东西可退。
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
    client = DeepSeekClient()
    collected: list[dict] = []
    executor = _make_tool_executor(collected)
    tools = build_chat_tools()
    # MOCK 模式判断该不该模拟工具调用时只看这段原文——不能把整个提示词丢进去，
    # 里面的语言名（「葡萄牙语」「西班牙语」）会被当成地名命中。
    probe = "\n".join([*(m.content for m in req.history), req.question])

    messages: list[dict] = [{"role": "user", "content": user}]
    reply = ""
    tool_results: list[dict] = []
    violations: list[Violation] = []

    for attempt in range(1 + _CHAT_VERIFY_RETRIES):
        outcome = await client.chat_with_tools(
            system,
            messages,
            tools,
            executor,
            language=req.language,
            tier=tier,
            purpose="chat",
            mock_probe=probe,
        )
        reply = (outcome.get("reply") or "").strip()
        tool_results += outcome.get("tool_results") or []
        sources = _sources_from(tool_results)

        violations = verify_chat_reply(reply, req.language, [s.url for s in sources])
        if not violations:
            logger.info(
                "追问对话完成，耗时 %.2fs（language=%s, tier=%s, history=%d 条，"
                "工具调用 %d 次，来源 %d 条）",
                time.perf_counter() - t0,
                req.language,
                tier,
                len(req.history),
                len(tool_results),
                len(sources),
            )
            return ChatResponse(reply=reply, sources=sources)

        _log_violations(f"追问回复第 {attempt + 1} 次", violations)
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": format_violations(violations, as_json=False)},
        ]

    logger.warning(
        "追问回复两次均未通过输出校验（%s），返回兜底文案",
        "；".join(v.code for v in violations),
    )
    return ChatResponse(reply=_UNRELIABLE_REPLY[req.language], sources=[])
