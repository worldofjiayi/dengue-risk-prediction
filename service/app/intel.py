"""Epidemiological intel tool: turns "what is dengue like in X" into one checkable lookup.

This is the only tool the chat model may call on its own (see
deepseek_client.chat_with_tools). It does two things, both traceable to a named source:

  1. Regional endemicity -- looks up the country/region table in
     app/data/dengue_endemicity.json (sources: WHO dengue fact sheet + CDC dengue risk
     map, 2026). This is a coarse-grained **travel background table**, not surveillance
     data, and it never takes part in any scoring.
  2. WHO Disease Outbreak News -- reads WHO's public OData endpoint live and keeps the
     entries whose title contains the target country; when there is no country-level entry
     it returns the latest global notices instead (their titles literally say "Global
     situation", so they cannot be misread as a notice about that country).

**Invariant: if we cannot find it, we say we cannot find it.** On a network failure the
result is lookup_failed=true with who_notices=[]; a WHO link is never made up from "general
knowledge". Links are always assembled from the UrlName the endpoint returned, with no way
for the model to interfere; the verifier (verifier.verify_chat_reply) then checks at the
exit that every link in the reply really came from this turn's tool result.

The WHO list is cached in-process for 12 hours: DON items are infrequent announcements, and
hitting who.int on every chat turn is both slow and impolite. The cache is injectable, and
tests use it to control time and the failure path.
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

# Function name exposed to the chat model (prompt, client and pipeline all share this constant)
INTEL_TOOL_NAME = "lookup_dengue_context"

# WHO Disease Outbreak News OData endpoint: public, no authentication needed
WHO_DON_API = (
    "https://www.who.int/api/news/diseaseoutbreaknews"
    "?$filter=contains(Title,'Dengue')&$orderby=PublicationDateAndTime desc"
)
WHO_ITEM_BASE = "https://www.who.int/emergencies/disease-outbreak-news/item/"
WHO_TIMEOUT = 8.0

# In-process cache lifetime: 12 hours
CACHE_TTL_SECONDS = 12 * 60 * 60
# How many notices one lookup returns at most
MAX_NOTICES = 3

_CJK_RE = re.compile(r"[㐀-䶿一-鿿]")

# WHO notices used in MOCK mode (**real, existing DON entries**, not made-up links).
# They go through exactly the same selection logic as in real mode, so the payload shape is
# field-for-field identical in both modes.
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
    """WHO notice fetch failed. Internal to this module; surfaces as lookup_failed=True."""


# ---------- Regional table ----------


@lru_cache(maxsize=1)
def load_endemicity() -> dict:
    """Read and cache dengue_endemicity.json."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def sources_note() -> dict:
    """Source statement for the regional table (WHO fact sheet + CDC map, 2026)."""
    return dict(load_endemicity().get("_sources", {}))


def _normalise_key(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


@lru_cache(maxsize=1)
def _alias_matchers() -> tuple[re.Pattern | None, tuple[str, ...]]:
    """Pre-compiled alias matchers: one big regex for Latin aliases, substring match for CJK.

    Both sides are sorted by descending length, so "south korea" matches before "korea" and
    "el salvador" before "salvador". Latin aliases are bounded by "not alphanumeric"
    look-around rather than \\b -- some aliases carry punctuation, such as "u.s." and
    "côte d'ivoire".
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
    """Find the first recognisable country/region in free text; return the canonical English name.

    Used in two places: as the fallback parse when the model passes a whole sentence in as
    location, and in MOCK mode to decide whether this turn should simulate a tool call.
    """
    if not text:
        return None
    aliases = load_endemicity()["aliases"]
    pattern, cjk = _alias_matchers()

    best: tuple[int, int, str] | None = None  # (length, -position, canonical name)
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
    """Resolve a place name from the user/model into (canonical English name, matched?)."""
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


# ---------- WHO notices ----------

# Module-level cache with a timestamp. items=None means "never fetched successfully yet".
_NOTICE_CACHE: dict = {"fetched_at": 0.0, "items": None}


def clear_notice_cache() -> None:
    """Clear the WHO notice cache (for tests)."""
    _NOTICE_CACHE["fetched_at"] = 0.0
    _NOTICE_CACHE["items"] = None


def seed_notice_cache(items: list[dict], fetched_at: float | None = None) -> None:
    """Write the cache directly (for tests: verify no request is sent within 12 hours)."""
    _NOTICE_CACHE["items"] = list(items)
    _NOTICE_CACHE["fetched_at"] = time.time() if fetched_at is None else fetched_at


def notice_cache_state() -> dict:
    """Read-only snapshot (for tests and troubleshooting)."""
    items = _NOTICE_CACHE["items"]
    return {
        "fetched_at": _NOTICE_CACHE["fetched_at"],
        "count": None if items is None else len(items),
    }


def fetch_who_notices() -> list[dict]:
    """Real network request: fetch WHO Disease Outbreak News entries with Dengue in the title."""
    try:
        with httpx.Client(timeout=WHO_TIMEOUT) as client:
            resp = client.get(WHO_DON_API, headers={"Accept": "application/json"})
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:  # every httpx exception + JSON parse errors
        raise IntelLookupError(
            f"The WHO Disease Outbreak News API is unavailable: {exc}"
        ) from exc
    items = payload.get("value") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise IntelLookupError("Malformed WHO API response: missing value list")
    return items


def _cached_notices(
    fetcher: Callable[[], list[dict]], now: float
) -> tuple[list[dict], bool]:
    """Return (raw items, lookup failed?). A cache hit sends no request."""
    cached = _NOTICE_CACHE["items"]
    if cached is not None and (now - _NOTICE_CACHE["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached, False
    try:
        items = fetcher()
    except Exception:
        # If we cannot get it, say so honestly -- never fall back to "this is probably the link"
        logger.warning(
            "Failed to fetch WHO Disease Outbreak News, no sources will be offered this round",
            exc_info=True,
        )
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
    """Pick the notices about the target country; with none, fall back to the latest global ones.

    Falling back to global notices is safe: their titles say "Global situation", which by
    itself states that they are not an announcement about any one country, so the model
    cannot mislead by citing them.
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


# ---------- Public tool function ----------


def lookup_dengue_context(
    location: str,
    *,
    now: float | None = None,
    fetcher: Callable[[], list[dict]] | None = None,
) -> dict:
    """Look up the dengue background for a place. This is the tool the chat model may call.

    The shape of the return value is fixed (identical in MOCK and real mode):
        location      canonical English name; the raw input if it was not recognised
        matched       whether the place name was recognised in the regional table
        endemicity    high | moderate | low | none | unknown
        season_note   short seasonal/geographic note; None when there is no match
        who_notices   <=3 items of {title, date, url}, newest publication first
        lookup_failed the WHO endpoint returned nothing this time (network failure);
                      who_notices is then necessarily empty
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
        "Intel lookup: location=%r -> %s (matched=%s, endemicity=%s, notices=%d, failed=%s)",
        raw,
        result["location"],
        matched,
        result["endemicity"],
        len(notices),
        failed,
    )
    return result
