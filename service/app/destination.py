"""Pre-travel destination lookup (POST /api/destination): dengue in a place, last three months.

Three layers; trustworthiness falls and cost rises as you go down:

  Layer 1  Built-in regional table + WHO Disease Outbreak News (app.intel)
           -- stable, free, and half of it still answers offline, so it **always runs first**.
  Layer 2  Web search (DeepSeek's Anthropic endpoint + the server-side web_search tool)
           -- the only layer that can answer "how have the last three months been", but it
           is billed per search and may come back with nothing at all.
  Layer 3  Pre-travel advice (deepseek_client.fallback_travel_advice)
           -- fixed copy, not model output, so it is always available and always compliant.

**This endpoint produces no score whatsoever.** Location never takes part in scoring, and it
never changes the exposure band. Returning a "destination risk score" here would mean making
up a number out of a coarse-grained country table.

Three invariants:
  1. recent_findings non-empty ⟺ search_status == "ok". If the search did not run, ran and
     found nothing, or its result failed the output check, this section is cleared -- better
     to say less than to say something we cannot stand behind.
  2. Every link in sources really does come from some endpoint: a WHO notice (origin=who) or
     a search result (origin=search). There is no third kind of source, and no "this is
     probably the link".
  3. The same (canonical place name, language) is searched only once within the TTL. Only a
     successful lookup is cached; failures are not, otherwise one network blip would stay
     pinned in the cache for 6 hours.
"""

import logging
import re
import time

from app.config import get_settings
from app.deepseek_client import DeepSeekClient, DeepSeekError, fallback_travel_advice
from app.eval_log import log_search
from app.intel import lookup_dengue_context, resolve_location
from app.prompt_builder import build_destination_prompt
from app.schemas import (
    DISCLAIMERS,
    MODEL_NOTES,
    DestinationAdvice,
    DestinationRequest,
    DestinationResponse,
    WhoNotice,
    merge_sources,
    select_search_sources,
)
from app.verifier import format_violations, verify_chat_reply

logger = logging.getLogger(__name__)

# How many recent_findings at most (the prompt asks for 2-4; this is the fallback truncation)
MAX_FINDINGS = 4
# Character cap for one finding: going over it means the model ignored the format,
# so the whole section is treated as unusable
MAX_FINDING_CHARS = 400
# Bullet markers for finding lines
_BULLET_PREFIXES = ("- ", "* ", "• ", "· ", "– ", "— ")
# Markdown bold/italic markers (in practice the model ignores the "no Markdown" instruction)
_EMPHASIS_RE = re.compile(r"\*{1,3}|__")

# How many re-asks the output check gets at most (first attempt not counted). A re-ask costs
# another search, so it only gets one, and the second attempt drops max_uses to 1 --
# rewording does not need another lookup.
_VERIFY_RETRIES = 1
_RETRY_MAX_USES = 1

# Endemicity values the regional table is allowed to return (matches schemas.Endemicity)
_ENDEMICITY_VALUES: tuple[str, ...] = ("high", "moderate", "low", "none", "unknown")


# ---------- Cache: (canonical place name, language) -> full response ----------
#
# The whole response is cached, not just the search result: on a hit not even the WHO
# endpoint is touched, so one request really does make zero external calls. The clock is
# injectable, so tests can simply push time forward.

_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}


def clear_destination_cache() -> None:
    """Clear the destination cache (for tests)."""
    _CACHE.clear()


def destination_cache_state() -> dict:
    """Read-only snapshot (for tests and troubleshooting)."""
    return {"size": len(_CACHE), "keys": sorted(_CACHE)}


def _cache_key(canonical: str, language: str) -> tuple[str, str]:
    return (canonical.strip().lower(), language)


def _cache_get(key: tuple[str, str], now: float) -> dict | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    stored_at, payload = entry
    ttl = get_settings().search_cache_ttl_seconds
    if ttl <= 0 or (now - stored_at) >= ttl:
        _CACHE.pop(key, None)
        return None
    return payload


def _cache_put(key: tuple[str, str], payload: dict, now: float) -> None:
    if get_settings().search_cache_ttl_seconds > 0:
        _CACHE[key] = (now, payload)


# ---------- Parsing and assembly ----------


def _strip_emphasis(text: str) -> str:
    """Strip Markdown bold/italic markers.

    The prompt already says "no Markdown", and in practice the model writes
    "- **Case counts stay low:** ..." anyway. The front end renders plain text, so the
    asterisks show up verbatim; rather than keep fighting the model, wipe them here --
    only the markers, not one character of the content.
    """
    return _EMPHASIS_RE.sub("", text or "").strip()


def parse_findings(reply: str) -> list[str]:
    """Split the model's prose reply into a list of findings.

    First look for lines beginning with "- " -- that is the format the prompt asks for.
    When there is not a single one, fall back to "every non-blank line is one finding",
    but **drop bare link lines** (sources are listed separately under sources and should
    not be mixed into the findings). Over-long items are simply discarded: they mean the
    model ignored the format, and one finding fewer beats a block of text nobody reads.
    """
    lines = [line.strip() for line in (reply or "").splitlines()]
    bullets: list[str] = []
    others: list[str] = []
    for line in lines:
        if not line:
            continue
        for prefix in _BULLET_PREFIXES:
            if line.startswith(prefix):
                bullets.append(line[len(prefix) :].strip())
                break
        else:
            others.append(line)

    chosen = [_strip_emphasis(t) for t in (bullets or others)]
    findings = [
        text
        for text in chosen
        if text and len(text) <= MAX_FINDING_CHARS and not text.lower().startswith("http")
    ]
    return findings[:MAX_FINDINGS]


def is_plausible_place(raw: str) -> bool:
    """Whether an input the regional table did not recognise still merits one search.

    What gets through is "a real place name the table does not list" (cities, states,
    regions); what gets stopped is anything plainly not a place name: too long, carrying a
    link, or obviously a whole sentence. Better to pay for the odd extra search than to
    degrade to "not found" over one city missing from the table -- but equally, a whole
    prompt injection must not become the search query.
    """
    text = (raw or "").strip()
    if not (2 <= len(text) <= 60):
        return False
    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered or "\n" in text:
        return False
    return len(text.split()) <= 6


# ---------- Main flow ----------


async def _search_once(
    client: DeepSeekClient,
    system: str,
    user: str,
    language: str,
    location: str,
    max_uses: int,
) -> dict:
    """Issue one search call: {"reply", "sources", "search_count"}. Failures propagate."""
    return await client.chat_anthropic_search(
        system,
        [{"role": "user", "content": user}],
        language=language,
        max_uses=max_uses,
        purpose="destination",
        mock_location=location,
    )


async def run_destination(
    req: DestinationRequest,
    *,
    now: float | None = None,
    client: DeepSeekClient | None = None,
) -> DestinationResponse:
    """Main pre-travel lookup flow. now is injectable (tests use it to advance cache time)."""
    settings = get_settings()
    t0 = time.perf_counter()
    clock = time.time() if now is None else now
    language = req.language

    canonical, matched = resolve_location(req.location)
    key = _cache_key(canonical, language)
    cached = _cache_get(key, clock)
    if cached is not None:
        logger.info("目的地查询命中缓存：%s / %s（未发起任何外部调用）", canonical, language)
        return DestinationResponse.model_validate(cached)

    # ---- Layer 1: regional table + WHO notices (free and stable, always runs first) ----
    intel_result = lookup_dengue_context(req.location, now=clock)
    display_location = str(intel_result.get("location") or canonical or req.location)
    endemicity = str(intel_result.get("endemicity") or "unknown")
    who_notices = list(intel_result.get("who_notices") or [])

    # ---- Layer 2: web search (the only layer that can answer "the last three months") ----
    findings: list[str] = []
    search_sources: list[dict] = []
    search_count = 0
    status = "disabled"

    should_search = settings.search_enabled and (
        matched or is_plausible_place(req.location)
    )
    if settings.search_enabled and not should_search:
        # The switch is on, this input just is not worth paying to search -- to the
        # outside world the same thing as "searched and found nothing"
        status = "degraded"
        logger.info("目的地查询：%r 不像地名，跳过检索", req.location[:60])

    if should_search:
        status = "degraded"
        system, user = build_destination_prompt(
            display_location, language, intel_result
        )
        deepseek = client or DeepSeekClient()
        max_uses = settings.search_max_uses
        prompt = user
        for attempt in range(1 + _VERIFY_RETRIES):
            try:
                outcome = await _search_once(
                    deepseek, system, prompt, language, display_location, max_uses
                )
            except DeepSeekError as exc:
                logger.warning("目的地检索失败（第 %d 次）：%s", attempt + 1, exc)
                break

            search_count += int(outcome.get("search_count") or 0)
            for item in outcome.get("sources") or []:
                if isinstance(item, dict):
                    search_sources.append(item)

            reply = outcome.get("reply") or ""
            candidate = parse_findings(reply)
            if not candidate:
                logger.info("目的地检索没有可用要点（location=%s）", display_location)
                break

            search_sources = select_search_sources(search_sources, reply)
            merged = merge_sources(who_notices, search_sources)
            allowed = [s.url for s in merged]
            violations = verify_chat_reply(
                "\n".join(candidate), language, allowed
            )
            if not violations:
                findings = candidate
                status = "ok" if merged else "degraded"
                break

            logger.warning(
                "目的地要点第 %d 次未通过输出校验：%s",
                attempt + 1,
                "；".join(v.code for v in violations),
            )
            # A re-ask does not need another lookup -- the facts are already in
            # the previous round's search results
            prompt = user + "\n\n" + format_violations(violations, as_json=False)
            max_uses = _RETRY_MAX_USES

    # Invariant: only "ok" carries findings. degraded/disabled are always cleared;
    # we do not serve text we are unsure of.
    if status != "ok":
        findings = []

    if endemicity not in _ENDEMICITY_VALUES:  # do not 500 if someone breaks the table
        logger.warning("地区表给出了未知的流行程度取值：%r", endemicity)
        endemicity = "unknown"

    merged_sources = merge_sources(
        who_notices, select_search_sources(search_sources, "\n".join(findings))
    )
    advice = fallback_travel_advice(language, endemicity)
    response = DestinationResponse(
        location=display_location,
        matched=bool(matched),
        endemicity=endemicity,
        season_note=intel_result.get("season_note"),
        who_notices=[
            WhoNotice(
                title=str(n.get("title") or ""),
                date=str(n.get("date") or ""),
                url=str(n.get("url") or ""),
            )
            for n in who_notices
        ],
        recent_findings=findings,
        sources=merged_sources,
        advice=DestinationAdvice(**advice),
        search_status=status,
        disclaimer=DISCLAIMERS[language],
        model_note=MODEL_NOTES[language],
    )

    if status == "ok":
        _cache_put(key, response.model_dump(), clock)

    log_search(
        "destination",
        language,
        display_location,
        search_count,
        status,
        matched=bool(matched),
    )
    logger.info(
        "目的地查询完成，耗时 %.2fs（location=%s, endemicity=%s, 检索 %d 次，"
        "要点 %d 条，来源 %d 条，status=%s）",
        time.perf_counter() - t0,
        display_location,
        endemicity,
        search_count,
        len(findings),
        len(merged_sources),
        status,
    )
    return response
