"""流行病学情报工具：把「某地登革热什么情况」这个问题变成一次可核查的查询。

这是聊天模型唯一能自主调用的工具（见 deepseek_client.chat_with_tools）。
它做两件事，两件都可追溯到具名来源：

  1. 地区流行程度 —— 查 app/data/dengue_endemicity.json 的国家/地区表
     （来源：WHO 登革热实况报道 + CDC 登革热风险地图，2026）。
     这是一张粗粒度的**旅行背景表**，不是监测数据，也绝不参与任何评分。
  2. WHO 疾病暴发新闻（Disease Outbreak News）—— 实时读 WHO 的公开 OData 接口，
     筛出标题里含目标国家的条目；没有国家级条目就返回最新的全球通报
     （它们的标题本身就写着 "Global situation"，不会被误读成针对该国的通报）。

**不变量：查不到就说查不到。** 网络失败时返回 lookup_failed=true 且
who_notices=[]，绝不用「常识」编一条 WHO 链接出来。链接一律由接口返回的
UrlName 拼成，模型无从插手；校验器（verifier.verify_chat_reply）再在出口处
核对回复里的每个链接确实来自本轮工具结果。

WHO 列表在进程内缓存 12 小时：DON 是低频发布的公告，每轮聊天都去打一次
who.int 既慢又不礼貌。缓存可注入，测试用它控制时间与失败路径。
"""

import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent / "data" / "dengue_endemicity.json"

# 暴露给聊天模型的函数名（提示词、客户端、流水线三处共用同一个常量）
INTEL_TOOL_NAME = "lookup_dengue_context"

# WHO 疾病暴发新闻（Disease Outbreak News）OData 接口：公开、无需鉴权
WHO_DON_API = (
    "https://www.who.int/api/news/diseaseoutbreaknews"
    "?$filter=contains(Title,'Dengue')&$orderby=PublicationDateAndTime desc"
)
WHO_ITEM_BASE = "https://www.who.int/emergencies/disease-outbreak-news/item/"
WHO_TIMEOUT = 8.0

# 进程内缓存有效期：12 小时
CACHE_TTL_SECONDS = 12 * 60 * 60
# 单次查询最多回传几条通报
MAX_NOTICES = 3

_CJK_RE = re.compile(r"[㐀-䶿一-鿿]")

# MOCK 模式下的 WHO 通报（**真实存在的 DON 条目**，不是编造的链接）。
# 走与真实模式完全相同的筛选逻辑，因此两种模式的 payload 形状逐字段一致。
MOCK_NOTICE_ITEMS: tuple[dict, ...] = (
    {
        "Title": "Dengue - Global situation",
        "PublicationDateAndTime": "2024-05-30T18:00:00Z",
        "UrlName": "2024-DON518",
    },
    {
        "Title": "Dengue - Bangladesh",
        "PublicationDateAndTime": "2023-08-11T11:52:45Z",
        "UrlName": "2023-DON481",
    },
    {
        "Title": "Dengue - the Region of the Americas",
        "PublicationDateAndTime": "2023-07-19T17:00:00Z",
        "UrlName": "2023-DON475",
    },
    {
        "Title": "Dengue- Global situation",
        "PublicationDateAndTime": "2023-12-21T19:00:19Z",
        "UrlName": "2023-DON498",
    },
    {
        "Title": "Dengue - Pakistan",
        "PublicationDateAndTime": "2022-10-13T18:00:00Z",
        "UrlName": "2022-DON414",
    },
)


class IntelLookupError(Exception):
    """WHO 通报拉取失败。只在模块内部使用，对外转成 lookup_failed=True。"""


# ---------- 地区表 ----------


@lru_cache(maxsize=1)
def load_endemicity() -> dict:
    """读取并缓存 dengue_endemicity.json。"""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def sources_note() -> dict:
    """地区表的来源声明（WHO 实况报道 + CDC 地图，2026）。"""
    return dict(load_endemicity().get("_sources", {}))


def _normalise_key(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@lru_cache(maxsize=1)
def _alias_matchers() -> tuple[re.Pattern | None, tuple[str, ...]]:
    """预编译别名匹配器：拉丁别名走一条大正则，CJK 别名走子串匹配。

    两边都按长度降序，保证 "south korea" 先于 "korea"、"el salvador" 先于
    "salvador" 命中。拉丁别名两侧用「非字母数字」前后瞻而不是 \\b——
    别名里有 "u.s."、"côte d'ivoire" 这种带标点的写法。
    """
    aliases = load_endemicity()["aliases"]
    latin = sorted((k for k in aliases if not _CJK_RE.search(k)), key=len, reverse=True)
    cjk = tuple(sorted((k for k in aliases if _CJK_RE.search(k)), key=len, reverse=True))
    pattern = None
    if latin:
        pattern = re.compile(
            r"(?<![0-9A-Za-z])(" + "|".join(re.escape(k) for k in latin) + r")(?![0-9A-Za-z])",
            re.IGNORECASE,
        )
    return pattern, cjk


def find_location(text: str) -> str | None:
    """在自由文本里找出第一个可识别的国家/地区，返回规范英文名。

    用于两处：模型把整句话当 location 传进来时的兜底解析，以及 MOCK 模式下
    判断这轮该不该模拟一次工具调用。
    """
    if not text:
        return None
    aliases = load_endemicity()["aliases"]
    pattern, cjk = _alias_matchers()

    best: tuple[int, int, str] | None = None  # (长度, -位置, 规范名)
    if pattern is not None:
        for match in pattern.finditer(text):
            key = _normalise_key(match.group(1))
            canonical = aliases.get(key)
            if canonical:
                candidate = (len(key), -match.start(), canonical)
                if best is None or candidate > best:
                    best = candidate
    lowered = text.lower()
    for key in cjk:
        index = lowered.find(key)
        if index >= 0:
            candidate = (len(key), -index, aliases[key])
            if best is None or candidate > best:
                best = candidate
    return best[2] if best else None


def resolve_location(location: str) -> tuple[str, bool]:
    """把用户/模型给的地名解析成 (规范英文名, 是否命中)。"""
    raw = (location or "").strip()
    if not raw:
        return "", False
    aliases = load_endemicity()["aliases"]
    key = _normalise_key(raw)
    if key in aliases:
        return aliases[key], True
    trimmed = key.strip(" .,!?;:'\"()[]，。！？、《》")
    if trimmed in aliases:
        return aliases[trimmed], True
    found = find_location(raw)
    if found:
        return found, True
    return raw, False


# ---------- WHO 通报 ----------

# 模块级带时间戳的缓存。items=None 表示「还没成功拉过」。
_NOTICE_CACHE: dict = {"fetched_at": 0.0, "items": None}


def clear_notice_cache() -> None:
    """清空 WHO 通报缓存（测试用）。"""
    _NOTICE_CACHE["fetched_at"] = 0.0
    _NOTICE_CACHE["items"] = None


def seed_notice_cache(items: list[dict], fetched_at: float | None = None) -> None:
    """直接写入缓存（测试用：验证 12 小时内不再发请求）。"""
    _NOTICE_CACHE["items"] = list(items)
    _NOTICE_CACHE["fetched_at"] = time.time() if fetched_at is None else fetched_at


def notice_cache_state() -> dict:
    """只读快照（测试与排障用）。"""
    items = _NOTICE_CACHE["items"]
    return {
        "fetched_at": _NOTICE_CACHE["fetched_at"],
        "count": None if items is None else len(items),
    }


def fetch_who_notices() -> list[dict]:
    """真实网络请求：拉 WHO 疾病暴发新闻里标题含 Dengue 的条目。"""
    try:
        with httpx.Client(timeout=WHO_TIMEOUT) as client:
            resp = client.get(WHO_DON_API, headers={"Accept": "application/json"})
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:  # httpx 各类异常 + JSON 解析异常
        raise IntelLookupError(f"WHO 疾病暴发新闻接口不可用：{exc}") from exc
    items = payload.get("value") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise IntelLookupError("WHO 接口返回结构异常：缺少 value 列表")
    return items


def _cached_notices(
    fetcher: Callable[[], list[dict]], now: float
) -> tuple[list[dict], bool]:
    """返回 (原始条目, 是否查询失败)。缓存命中就不发请求。"""
    cached = _NOTICE_CACHE["items"]
    if cached is not None and (now - _NOTICE_CACHE["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached, False
    try:
        items = fetcher()
    except Exception:
        # 拿不到就诚实地说拿不到——绝不退回「大概是这个链接」
        logger.warning("WHO 疾病暴发新闻拉取失败，本轮不提供任何来源", exc_info=True)
        return [], True
    _NOTICE_CACHE["items"] = items
    _NOTICE_CACHE["fetched_at"] = now
    return items, False


def _publication_key(item: dict) -> str:
    return str(item.get("PublicationDateAndTime") or item.get("PublicationDate") or "")


def _to_notice(item: dict) -> dict | None:
    url_name = str(item.get("UrlName") or "").strip()
    title = " ".join(str(item.get("Title") or "").split())
    if not url_name or not title:
        return None
    return {
        "title": title,
        "date": _publication_key(item)[:10],
        "url": WHO_ITEM_BASE + url_name,
    }


def select_notices(items: list[dict], canonical: str | None) -> list[dict]:
    """挑出与目标国家相关的通报；没有就退回最新的全球通报。

    退回全球通报是安全的：它们的标题写着 "Global situation"，本身就说明
    自己不是针对某个国家的公告，模型引用时不会造成误导。
    """
    ordered = sorted(items, key=_publication_key, reverse=True)
    chosen: list[dict] = []
    if canonical:
        needle = canonical.lower()
        chosen = [i for i in ordered if needle in str(i.get("Title", "")).lower()]
    if not chosen:
        chosen = [i for i in ordered if "global" in str(i.get("Title", "")).lower()] or ordered
    notices = [n for n in (_to_notice(i) for i in chosen[: MAX_NOTICES * 2]) if n]
    return notices[:MAX_NOTICES]


# ---------- 对外工具函数 ----------


def lookup_dengue_context(
    location: str,
    *,
    now: float | None = None,
    fetcher: Callable[[], list[dict]] | None = None,
) -> dict:
    """查询某地的登革热背景。这就是聊天模型可以调用的那个工具。

    返回固定形状（MOCK 与真实模式完全一致）：
        location      规范英文名；没认出来就是原样输入
        matched       是否在地区表里认出了这个地名
        endemicity    high | moderate | low | none | unknown
        season_note   简短的季节/地域说明；未命中为 None
        who_notices   ≤3 条 {title, date, url}，按发布时间倒序
        lookup_failed WHO 接口这次没拉到（网络失败），who_notices 必为空
    """
    raw = (location or "").strip()
    canonical, matched = resolve_location(raw)
    entry = load_endemicity()["countries"].get(canonical) if matched else None

    if fetcher is not None:
        items, failed = _cached_notices(fetcher, time.time() if now is None else now)
    elif get_settings().mock_mode:
        items, failed = list(MOCK_NOTICE_ITEMS), False
    else:
        items, failed = _cached_notices(
            fetch_who_notices, time.time() if now is None else now
        )

    notices = select_notices(items, canonical if matched else None)
    result = {
        "location": canonical if matched else (raw or "unknown"),
        "matched": matched,
        "endemicity": entry["level"] if entry else "unknown",
        "season_note": entry.get("season") if entry else None,
        "who_notices": notices,
        "lookup_failed": failed,
    }
    logger.info(
        "情报查询：location=%r -> %s（matched=%s, endemicity=%s, notices=%d, failed=%s）",
        raw,
        result["location"],
        matched,
        result["endemicity"],
        len(notices),
        failed,
    )
    return result
