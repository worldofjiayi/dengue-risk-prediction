"""行前目的地查询（POST /api/destination）：某地最近三个月的登革热情况。

三层结构，可信度与成本都从上到下递减/递增：

  第 1 层  内置地区表 + WHO 疾病暴发新闻（app.intel）
           —— 稳定、免费、离线也有一半能答，**永远先跑**。
  第 2 层  联网检索（DeepSeek 的 Anthropic 端点 + web_search 服务端工具）
           —— 只有它能回答「最近三个月怎么样」，但按次计费且可能一无所获。
  第 3 层  出行前建议（deepseek_client.fallback_travel_advice）
           —— 固定文案，不来自模型，因此永远可用、永远合规。

**这个接口不产出任何评分。** 地点从来不参与打分，也不改变暴露档位。要在这里
返回一个「目的地风险分」，就只能拿一张粗粒度国家表编一个数字出来。

三条不变量：
  1. recent_findings 非空 ⟺ search_status == "ok"。检索没跑、跑了没结果、
     或者结果没通过出口校验，这一段一律清空——宁可少说，不可说不准的。
  2. sources 里的每条链接都真的来自某个接口：WHO 通报（origin=who）或
     检索结果（origin=search）。没有第三种来源，也没有「大概是这个链接」。
  3. 同一个 (规范地名, 语言) 在 TTL 内只检索一次。缓存的只有成功的那次；
     失败不缓存，否则一次网络抖动会被钉在缓存里 6 小时。
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

# recent_findings 最多几条（提示词要求 2-4 条，这里是兜底截断）
MAX_FINDINGS = 4
# 单条要点的字符上限：超过就是模型没按格式走，整段当作不可用
MAX_FINDING_CHARS = 400
# 要点行的项目符号
_BULLET_PREFIXES = ("- ", "* ", "• ", "· ", "– ", "— ")
# Markdown 的加粗/斜体记号（实测模型会无视「不要用 Markdown」这条要求）
_EMPHASIS_RE = re.compile(r"\*{1,3}|__")

# 出口校验最多重问几次（不含首次）。重问会再花一次检索，所以只给一次，
# 而且第二次把 max_uses 压到 1——重写措辞不需要再查一遍。
_VERIFY_RETRIES = 1
_RETRY_MAX_USES = 1

# 地区表允许的流行程度取值（与 schemas.Endemicity 一致）
_ENDEMICITY_VALUES: tuple[str, ...] = ("high", "moderate", "low", "none", "unknown")


# ---------- 缓存：(规范地名, 语言) -> 完整响应 ----------
#
# 缓存整份响应而不是只缓存检索结果：命中时连 WHO 接口都不用碰，
# 一次请求真正做到零外部调用。时钟可注入，测试直接推时间。

_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}


def clear_destination_cache() -> None:
    """清空目的地缓存（测试用）。"""
    _CACHE.clear()


def destination_cache_state() -> dict:
    """只读快照（测试与排障用）。"""
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


# ---------- 解析与组装 ----------


def _strip_emphasis(text: str) -> str:
    """去掉 Markdown 的加粗/斜体记号。

    提示词已经写了「不要用 Markdown」，实测模型照样会写
    「- **病例数维持低位：** …」。前端渲染的是纯文本，星号会原样露出来，
    与其和模型拉锯，不如在这里擦掉——只擦记号，不动一个字的内容。
    """
    return _EMPHASIS_RE.sub("", text or "").strip()


def parse_findings(reply: str) -> list[str]:
    """把模型的散文回复拆成要点列表。

    先找「- 」开头的行——提示词要的就是这个格式。一条都没有时退回「每个非空行
    算一条」，但**丢掉纯链接行**（来源由 sources 单独列出，不该混进要点里）。
    超长条目直接丢弃：那说明模型没按格式走，与其塞一段没人读的文字，不如少一条。
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
    """地区表没认出来的输入，还值不值得为它花一次检索。

    放行的是「表里没有的真地名」（城市、州、地区），拦下的是明显不是地名的东西：
    太长、带链接、或者一看就是一整句话。宁可偶尔多查一次，也不要因为一个
    未收录的城市名就退化成「查不到」——但也不能让整段提示注入变成检索关键词。
    """
    text = (raw or "").strip()
    if not (2 <= len(text) <= 60):
        return False
    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered or "\n" in text:
        return False
    return len(text.split()) <= 6


# ---------- 主流程 ----------


async def _search_once(
    client: DeepSeekClient,
    system: str,
    user: str,
    language: str,
    location: str,
    max_uses: int,
) -> dict:
    """发一次检索调用，返回 {"reply", "sources", "search_count"}。失败向上抛。"""
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
    """行前查询主流程。now 可注入（测试用来推缓存时间）。"""
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

    # ---- 第 1 层：地区表 + WHO 通报（免费且稳定，永远先跑）----
    intel_result = lookup_dengue_context(req.location, now=clock)
    display_location = str(intel_result.get("location") or canonical or req.location)
    endemicity = str(intel_result.get("endemicity") or "unknown")
    who_notices = list(intel_result.get("who_notices") or [])

    # ---- 第 2 层：联网检索（只有它能回答「最近三个月」）----
    findings: list[str] = []
    search_sources: list[dict] = []
    search_count = 0
    status = "disabled"

    should_search = settings.search_enabled and (
        matched or is_plausible_place(req.location)
    )
    if settings.search_enabled and not should_search:
        # 开关是开的，只是这个输入不值得花钱查——对外与「查了没查到」同义
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
            # 重问时不需要再查一遍——事实已经在上一轮的检索结果里
            prompt = user + "\n\n" + format_violations(violations, as_json=False)
            max_uses = _RETRY_MAX_USES

    # 不变量：只有 ok 才带要点。degraded/disabled 一律清空，不端不确定的文本。
    if status != "ok":
        findings = []

    if endemicity not in _ENDEMICITY_VALUES:  # 地区表被人改坏时也不要 500
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
