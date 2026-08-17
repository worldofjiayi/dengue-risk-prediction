"""Assessment pipeline: questionnaire -> feature encoding -> three-model scoring ->
DeepSeek advice generation -> result assembly.

Feature encoding is **deterministic** (app.ml_model.encode_features) and is the
authoritative source. The first DeepSeek call does one supplementary job only: pick out
the symptoms the user "described but did not tick" in their free-text notes, and lift
those items from unknown to yes. A failure in that step does not affect the
assessment -- we simply keep the deterministic encoding.

Besides the model scores, this module produces two **rule-based judgements that have
nothing to do with the model**:
  warning_signs    -- WHO dengue warning signs (VOMITO / PETEQUIA_N)
  exposure_context -- epidemiological exposure background (a confirmed case nearby /
                      travel to an outbreak area / a cluster of fevers around the user)
Neither enters the 26-dimensional feature vector nor takes part in scoring; they are
only presented alongside the scores.

This module also carries the /api/chat follow-up conversation (run_chat) -- stateless,
with the context posted back by the front end, and **split into two paths by whether the
question mentions a location**: with no location the model is handed a tool it can call
on its own (app.intel.lookup_dengue_context); with a location we switch to web search
(see _run_chat_with_search). The pre-travel destination lookup /api/destination lives in
app.destination; the two share the same source and verification rules.

**There is one more gate after generation**: every LLM text has to pass app.verifier's
rule checks before it is returned (dosage, infection probability, urgency of seeking
care, language consistency, structure, fabricated links). If it fails we ask again once
with the violations spelled out; if it still fails we fall back to the template /
fallback copy. Therefore:

  - A failure to generate advice **no longer fails the whole assessment**. The scores are
    computed locally and are the part of this service that is genuinely worth something;
    turning a 200 into a 502 because one natural-language sentence could not be fetched
    pays for a replaceable paragraph with the result the user already has. We now return
    200 + template advice + advice_source=template.
  - /api/chat has nothing to fall back on (the reply is the entire output), so it still
    returns 502.

The raw notes text is never written to the log, to keep sensitive user information out
of the logs.
"""

import logging
import time

from app.config import get_settings
from app.deepseek_client import DeepSeekClient, DeepSeekError, fallback_advice
from app.eval_log import log_assessment, log_search
from app.intel import INTEL_TOOL_NAME, find_location, lookup_dengue_context
from app.ml_model import encode_features, get_epi_week, get_model
from app.prompt_builder import (
    build_advice_prompt,
    build_chat_prompt,
    build_chat_search_prompt,
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
    merge_sources,
    select_search_sources,
)
from app.verifier import (
    Violation,
    format_violations,
    verify_advice,
    verify_chat_reply,
)

logger = logging.getLogger(__name__)

# Risk levels from low to high, used to take "the highest band of the three models"
_LEVEL_ORDER: tuple[str, ...] = ("low", "medium", "high")

# Fallback copy for a missing summary (five languages)
_FALLBACK_SUMMARY = {
    "zh-CN": "已完成风险评估，请结合下方建议做好防蚊防护与健康监测。",
    "zh-TW": "已完成風險評估，請結合下方建議做好防蚊防護與健康監測。",
    "en": "Assessment complete. Please follow the guidance below on mosquito protection and monitoring.",
    "es": "Evaluación completada. Siga las recomendaciones sobre protección contra mosquitos y vigilancia.",
    "pt": "Avaliação concluída. Siga as orientações abaixo sobre proteção contra mosquitos e monitoramento.",
}


# Fallback for a follow-up reply that failed output verification twice (five languages).
#
# Deliberately does not try to "answer roughly anyway": a failed check means the text
# contains something that should not be there, and serving it is worse than not
# answering -- better to admit this round is unreliable.
# "Empty reply" is now a violation too (the verifier's empty rule), so an empty reply
# shares this same path with every other violation; we no longer need a second fallback
# sentence, worded differently, that would sooner or later drift away from this one.
_UNRELIABLE_REPLY = {
    "zh-CN": "抱歉，我这次无法给出可靠的回答，请咨询当地的医疗机构或公共卫生服务。若症状加重请尽快就医。",
    "zh-TW": "抱歉，這次無法給出可靠的回覆，請諮詢當地醫療院所或公共衛生服務。若症狀加重請儘快就醫。",
    "en": "I can't produce a reliable answer right now — please consult a local health service. If your symptoms worsen, seek medical care promptly.",
    "es": "No puedo dar una respuesta fiable en este momento; consulte a un servicio de salud local. Si sus síntomas empeoran, busque atención médica lo antes posible.",
    "pt": "Não consigo dar uma resposta confiável agora — consulte um serviço de saúde local. Se os sintomas piorarem, procure atendimento médico o quanto antes.",
}

# How many re-asks output verification of the advice gets at most (first try excluded)
_ADVICE_VERIFY_RETRIES = 1
# How many re-asks output verification of a follow-up reply gets (first try excluded)
_CHAT_VERIFY_RETRIES = 1


def overall_tier(levels: list[str]) -> str:
    """Take the highest band out of a set of risk levels (high > medium > low).

    Gives advice generation and the follow-up conversation a single "overall band": as
    soon as any one model reports high, the overall wording follows high -- better to
    warn once too often than to let a high score be averaged away by two low ones.
    """
    best = "low"
    for level in levels:
        if level in _LEVEL_ORDER and _LEVEL_ORDER.index(level) > _LEVEL_ORDER.index(best):
            best = level
    return best


def evaluate_exposure(form: FormInput) -> ExposureContext:
    """**Rule-based judgement** of epidemiological exposure (no model involved).

        high   -- CONFIRMED_CASE or OUTBREAK_TRAVEL is yes
        medium -- FEVER_CLUSTER is yes and high was not reached
        low    -- everything else

    Only an explicit yes counts: as with symptom encoding, "don't know" does not mean
    "yes", and a user's uncertainty must not be used to raise the risk warning.

    Why this is not folded into the model: the SINAN notification data does not contain
    these three variables, the logistic regression has no coefficients for them, and any
    weight would be nothing but a made-up number. Presented separately, both users and
    clinicians can see which part comes from fitting data and which part from
    epidemiological common sense.
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
    """Use DeepSeek to add symptoms from the notes; any failure returns the original form."""
    system, user = build_feature_prompt(form)
    try:
        raw = await client.chat_json(system, user, purpose="features")
    except DeepSeekError:
        logger.warning(
            "Symptom extraction from the notes failed, keeping the deterministic encoding",
            exc_info=True,
        )
        return form

    infer = raw.get("infer")
    if not isinstance(infer, dict) or not infer:
        return form

    updated = dict(form.symptoms)
    applied: list[str] = []
    for code, value in infer.items():
        # Only lift "not yet answered" to "yes"; never overturn an explicit user answer
        if code in SYMPTOM_CODES and value == "yes" and updated.get(code) == "unknown":
            updated[code] = "yes"
            applied.append(code)

    if not applied:
        return form
    logger.info("Extra symptoms recognised in the notes: %s", applied)
    return form.model_copy(update={"symptoms": updated})


def _parse_advice(raw: dict, language: str) -> tuple[Advice, str]:
    """Parse the model's JSON into (Advice, summary). Raises DeepSeekError if malformed."""
    try:
        advice = Advice.model_validate(raw.get("advice", {}))
    except Exception as exc:  # pydantic ValidationError and other structural failures
        raise DeepSeekError("DeepSeek advice output does not meet the contract") from exc
    summary = str(raw.get("summary", "")).strip() or _FALLBACK_SUMMARY[language]
    return advice, summary


def _template_advice(language: str, tier: str) -> tuple[Advice, str]:
    """Fallback: shares the same per-band template as the MOCK demo
    (deepseek_client.fallback_advice).
    """
    raw = fallback_advice(language, tier)
    return Advice.model_validate(raw["advice"]), raw["summary"]


def _log_violations(where: str, violations: list[Violation]) -> None:
    logger.warning(
        "%s failed output verification: %s", where, "; ".join(v.code for v in violations)
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
    """Generate advice and make sure it passes output verification.

    Returns (advice, summary, advice_source).

    Real mode: generate -> verify -> re-ask once with the violations spelled out ->
    verify again -> fall back to the template if it still does not pass.
    MOCK mode: use the template directly, but **run the verification all the same** --
    the checks are pure rules and cost nothing, and the template is exactly what gets
    served to the user when real mode fails, so it must always be clean.
    This path is guarded by the "5 languages × 3 bands × 0 violations" test in tests.
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
        if violations:  # Should not happen: broken template copy, shout about it in the log
            _log_violations("MOCK template advice", violations)
        return advice, summary, "template"

    user_prompt = adv_user
    for attempt in range(1 + _ADVICE_VERIFY_RETRIES):
        try:
            raw = await client.chat_json(
                adv_system, user_prompt, purpose="advice", language=language, tier=tier
            )
            advice, summary = _parse_advice(raw, language)
        except DeepSeekError as exc:
            logger.error("Advice generation failed (attempt %d): %s", attempt + 1, exc)
            break

        violations = verify_advice(advice, summary, language, tier, warning_signs)
        if not violations:
            return advice, summary, "llm"
        _log_violations(f"Advice generation attempt {attempt + 1}", violations)
        # Splice the violations back into the prompt so the model only fixes those spots
        user_prompt = adv_user + "\n\n" + format_violations(violations, as_json=True)

    logger.warning("Advice fell back to the template copy (language=%s, tier=%s)", language, tier)
    advice, summary = _template_advice(language, tier)
    return advice, summary, "template"


async def run_assessment(form: FormInput) -> AssessmentResult:
    """Full assessment flow.

    A failure at any step raises, and the route above turns it into an HTTP error.
    """
    settings = get_settings()
    client = DeepSeekClient()
    t0 = time.perf_counter()

    # ---- Step 1: deterministic feature encoding ----
    # When there are notes and we are not in MOCK, let DeepSeek add symptoms from them first
    if form.notes.strip() and not settings.mock_mode:
        form = await _infer_notes_symptoms(form, client)
    features = encode_features(form)
    epi_week = get_epi_week()
    logger.info(
        "Step 1 feature encoding finished in %.2fs, epi_week=%d",
        time.perf_counter() - t0,
        epi_week,
    )

    # WHO warning signs: a rule-based judgement, independent of the model scores
    # (see schemas.WARNING_SIGN_CODES)
    warning_signs = [c for c in WARNING_SIGN_CODES if form.symptoms.get(c) == "yes"]
    if warning_signs:
        logger.info("User reported WHO warning signs: %s", warning_signs)

    # Epidemiological exposure: also a rule-based judgement, and it does not enter the
    # feature vector (see evaluate_exposure)
    exposure = evaluate_exposure(form)
    if exposure.factors:
        logger.info("Epidemiological exposure: level=%s factors=%s", exposure.level, exposure.factors)

    # ---- Step 2: score with the three models ----
    t1 = time.perf_counter()
    model = get_model()
    scores = model.score_all(features)
    explanations = model.explain_all(features)
    tier = overall_tier([s.level for s in scores.values()])
    logger.info(
        "Step 2 model scoring finished in %.2fs, A=%.1f(%s) B=%.1f(%s) B2=%.1f(%s), overall tier=%s",
        time.perf_counter() - t1,
        scores["A"].score, scores["A"].level,
        scores["B"].score, scores["B"].level,
        scores["B2"].score, scores["B2"].level,
        tier,
    )

    # ---- Step 3: DeepSeek advice generation (generate -> verify -> re-ask -> fallback) ----
    t2 = time.perf_counter()
    advice, summary, advice_source = await _produce_advice(
        form, scores, epi_week, warning_signs, exposure, tier, client, settings
    )
    logger.info(
        "Step 3 advice generation finished in %.2fs, source=%s",
        time.perf_counter() - t2,
        advice_source,
    )

    # ---- Step 4: assemble the result ----
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
    logger.info("Assessment finished, total %.2fs", time.perf_counter() - t0)

    log_assessment(form, features, scores, epi_week, exposure)
    return result


def _make_tool_executor(collected: list[dict]):
    """Build the tool executor that gets injected into the client.

    The client only knows that "there is a function it can call"; what the tool actually
    looks up, how the arguments are sanitised and where the results are stored all live
    here -- keeping the transport layer apart from the domain logic is what makes the
    client independently testable.
    """

    def execute(name: str, args: dict) -> dict:
        if name != INTEL_TOOL_NAME:
            logger.warning("The model requested an unknown tool: %r", name)
            return {"error": f"unknown tool '{name}'", "lookup_failed": True}
        location = str((args or {}).get("location", "")).strip()[:120]
        result = lookup_dengue_context(location)
        collected.append(result)
        return result

    return execute


def _sources_from(tool_results: list[dict]) -> list[Source]:
    """Gather WHO notices from the tool results into sources (deduped by url, order kept)."""
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
                    origin="who",
                    authority="official",  # a WHO notice is an official source by definition
                )
            )
    return sources


async def _run_chat_with_search(
    req: ChatRequest, location: str, t0: float
) -> ChatResponse:
    """**Search version** of the follow-up conversation: when a location shows up in the
    question, go online for the situation over the last three months.

    The difference from the function-tool path is not just "search has been added": here
    the model gets **no function tools at all**. The regional background (endemicity /
    season / WHO notices) is looked up by us and placed into the prompt -- that lookup is
    a local table plus a public endpoint cached for 12 hours, costs next to nothing, and
    yet gives the search a checkable foundation while also adding the WHO links to this
    round's citation allow-list.

    The allow-list is therefore the **union of two layers**: the links returned by the WHO
    tool plus the links the search really returned. Any third kind of link appearing in
    the reply is still fabricated_url, still gets one re-ask and then the fallback.
    """
    settings = get_settings()
    intel_result = lookup_dengue_context(location)
    who_notices = list(intel_result.get("who_notices") or [])

    system, user = build_chat_search_prompt(req, intel_result)
    client = DeepSeekClient()
    messages: list[dict] = [{"role": "user", "content": user}]
    max_uses = settings.search_max_uses
    search_count = 0
    reply = ""
    search_sources: list[dict] = []
    violations: list[Violation] = []

    for attempt in range(1 + _CHAT_VERIFY_RETRIES):
        outcome = await client.chat_anthropic_search(
            system,
            messages,
            language=req.language,
            max_uses=max_uses,
            purpose="chat",
            mock_location=location,
        )
        reply = (outcome.get("reply") or "").strip()
        search_count += int(outcome.get("search_count") or 0)
        search_sources += [s for s in (outcome.get("sources") or []) if isinstance(s, dict)]
        sources = merge_sources(
            who_notices, select_search_sources(search_sources, reply)
        )

        violations = verify_chat_reply(reply, req.language, [s.url for s in sources])
        if not violations:
            logger.info(
                "Follow-up chat (search version) finished in %.2fs (language=%s, location=%s, "
                "%d search(es), %d source(s))",
                time.perf_counter() - t0,
                req.language,
                location,
                search_count,
                len(sources),
            )
            log_search("chat", req.language, location, search_count, "ok", matched=True)
            return ChatResponse(reply=reply, sources=sources, search_count=search_count)

        _log_violations(f"Follow-up reply (search version) attempt {attempt + 1}", violations)
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": format_violations(violations, as_json=False)},
        ]
        # Rewording needs no second search: the facts are already in the previous round
        max_uses = 0

    logger.warning(
        "Follow-up reply (search version) failed output verification both times (%s), "
        "returning the fallback copy",
        "; ".join(v.code for v in violations),
    )
    log_search("chat", req.language, location, search_count, "degraded", matched=True)
    return ChatResponse(
        reply=_UNRELIABLE_REPLY[req.language], sources=[], search_count=search_count
    )


async def run_chat(req: ChatRequest) -> ChatResponse:
    """Follow-up conversation: stateless, context and history all posted back by the front end.

    **Two paths, decided by "is there a location in the question"** (see
    _run_chat_with_search):

      location + SEARCH_ENABLED -- use web search, so the answer can carry the situation
                                   over the last three months;
      everything else           -- use the original OpenAI function-calling path, which
                                   **spends not a cent on search**.

    This split is deliberate cost control. Search is billed per call and the number of
    calls is decided by the model (measured: one ordinary question triggered 4 searches
    and about 13.9k input tokens), whereas a question like "what does my score mean"
    needs no web access at all. Location recognition uses a local alias table
    (app.intel.find_location): zero cost, and it accepts Chinese, Spanish and Portuguese
    spellings.

    Both paths leave through the same gate: every reply passes verify_chat_reply before
    it is returned, and allowed_urls contains **only the links actually fetched in this
    round**. If verification fails we re-ask once with the violations spelled out; if it
    fails again we switch to the localised fallback sentence.

    Failures (DeepSeekError) propagate upward and the route turns them into a 502 -- chat
    has nothing else to fall back on.
    """
    t0 = time.perf_counter()
    tier = overall_tier(
        [
            block.level
            for block in (req.context.dengue, req.context.worsening, req.context.severe)
            if block is not None
        ]
    )
    # Location recognition looks only at the user's own text (this round's question + the
    # last few history entries) -- we must not throw the whole prompt in: the language
    # names inside it ("葡萄牙语" contains 葡萄牙/Portugal, "西班牙语" contains 西班牙/Spain)
    # would be matched as place names.
    probe = "\n".join([*(m.content for m in req.history), req.question])
    location = find_location(probe)
    settings = get_settings()
    if location and settings.search_enabled:
        return await _run_chat_with_search(req, location, t0)
    if location:
        logger.info(
        "Recognised location %s, but SEARCH_ENABLED=false, taking the no-search path", location
    )

    system, user = build_chat_prompt(req)
    client = DeepSeekClient()
    collected: list[dict] = []
    executor = _make_tool_executor(collected)
    tools = build_chat_tools()

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
                "Follow-up chat finished in %.2fs (language=%s, tier=%s, history=%d message(s), "
                "%d tool call(s), %d source(s))",
                time.perf_counter() - t0,
                req.language,
                tier,
                len(req.history),
                len(tool_results),
                len(sources),
            )
            return ChatResponse(reply=reply, sources=sources)

        _log_violations(f"Follow-up reply attempt {attempt + 1}", violations)
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": format_violations(violations, as_json=False)},
        ]

    logger.warning(
        "Follow-up reply failed output verification both times (%s), returning the fallback copy",
        "; ".join(v.code for v in violations),
    )
    return ChatResponse(reply=_UNRELIABLE_REPLY[req.language], sources=[])
