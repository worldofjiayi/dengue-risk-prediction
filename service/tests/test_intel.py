"""流行病学情报工具（app/intel.py）测试。

三件事必须钉死：
  1. 别名解析——五种语言的地名写法、常见变体，都要落到同一个规范名上；
  2. 缓存——12 小时内不许再打 who.int，过期后必须重拉；
  3. **诚实**——网络失败时返回 lookup_failed=true 且 who_notices=[]，
     绝不退回「大概是这个链接」。这条是整个工具能被信任的前提。

所有测试都不发真实网络请求：要么在 MOCK 模式下走内置数据，要么注入 fetcher /
替换 httpx.Client。
"""

import json
from pathlib import Path

import httpx
import pytest

from app import intel

SERVICE_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = SERVICE_ROOT / "app" / "data" / "dengue_endemicity.json"

VALID_LEVELS = {"high", "moderate", "low", "none"}

FAKE_ITEMS = [
    {
        "Title": "Dengue - Global situation",
        "PublicationDateAndTime": "2025-05-30T18:00:00Z",
        "UrlName": "2025-DON999",
    },
    {
        "Title": "Dengue - Brazil",
        "PublicationDateAndTime": "2025-03-01T10:00:00Z",
        "UrlName": "2025-DON900",
    },
    {
        "Title": "Dengue- Global situation",
        "PublicationDateAndTime": "2023-12-21T19:00:19Z",
        "UrlName": "2023-DON498",
    },
]


@pytest.fixture(autouse=True)
def _clean_cache():
    """每条测试都从空缓存开始，也不给下一条留残留。"""
    intel.clear_notice_cache()
    yield
    intel.clear_notice_cache()


@pytest.fixture()
def mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def live_mode(monkeypatch):
    """关掉 MOCK：走真实分支，但网络永远由测试自己替换掉。"""
    monkeypatch.setenv("MOCK_MODE", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------- 数据文件本身 ----------


def test_data_file_shape_and_coverage():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    countries = data["countries"]
    assert len(countries) >= 50
    for name, entry in countries.items():
        assert entry["level"] in VALID_LEVELS, name
        assert entry["season"].strip(), name
    # 覆盖到高流行带、中等区和常见客源地三类
    assert countries["Brazil"]["level"] == "high"
    assert countries["Indonesia"]["level"] == "high"
    assert countries["Taiwan"]["level"] == "moderate"
    assert countries["United Kingdom"]["level"] == "none"
    assert {countries[c]["level"] for c in countries} == VALID_LEVELS


def test_sources_name_who_and_cdc_for_2026():
    sources = intel.sources_note()
    assert sources["year"] == 2026
    joined = " ".join(sources["references"])
    assert "WHO" in joined and "fact sheet" in joined
    assert "CDC" in joined and "map" in joined


def test_every_alias_points_at_a_real_country():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    unknown = sorted(set(data["aliases"].values()) - set(data["countries"]))
    assert unknown == []


def test_sub_national_notes_where_they_matter():
    """粗到国家一级会误导的地方，必须在 season 注记里说清楚。"""
    countries = json.loads(DATA_PATH.read_text(encoding="utf-8"))["countries"]
    assert "Guangdong" in countries["China"]["season"]
    assert "Yunnan" in countries["China"]["season"]
    assert "Florida" in countries["United States"]["season"]
    assert "Queensland" in countries["Australia"]["season"]
    assert countries["Northern China"]["level"] == "none"


# ---------- 别名解析 ----------


@pytest.mark.parametrize(
    ("written", "canonical"),
    [
        ("Singapore", "Singapore"), ("新加坡", "Singapore"), ("Singapura", "Singapore"),
        ("Brazil", "Brazil"), ("Brasil", "Brazil"), ("巴西", "Brazil"),
        ("Thailand", "Thailand"), ("泰国", "Thailand"), ("泰國", "Thailand"),
        ("Tailandia", "Thailand"), ("Tailândia", "Thailand"),
        ("USA", "United States"), ("美国", "United States"), ("美國", "United States"),
        ("Estados Unidos", "United States"), ("u.s.", "United States"),
        ("México", "Mexico"), ("Filipinas", "Philippines"), ("Camboja", "Cambodia"),
        ("台灣", "Taiwan"), ("台湾", "Taiwan"),
        ("北京", "Northern China"), ("中国", "China"), ("廣東", "China"),
        ("  singapore  ", "Singapore"), ("SINGAPORE", "Singapore"),
    ],
)
def test_alias_resolution(written, canonical):
    got, matched = intel.resolve_location(written)
    assert (got, matched) == (canonical, True)


def test_unknown_location_is_not_matched():
    got, matched = intel.resolve_location("Narnia")
    assert (got, matched) == ("Narnia", False)
    assert intel.resolve_location("") == ("", False)


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("I am flying to Singapore next month.", "Singapore"),
        ("下个月我要去新加坡出差。", "Singapore"),
        ("Voy a viajar a Brasil en enero.", "Brazil"),
        ("Vou para a Tailândia na semana que vem.", "Thailand"),
        ("我需要现在去医院吗？", None),
        ("How does dengue spread?", None),
        ("Should I worry about my score?", None),
    ],
)
def test_find_location_in_free_text(sentence, expected):
    assert intel.find_location(sentence) == expected


def test_longest_alias_wins():
    """"south korea" 必须压过 "korea"，"el salvador" 压过 "salvador"。"""
    assert intel.find_location("moving to South Korea") == "South Korea"
    assert intel.find_location("a trip to El Salvador") == "El Salvador"


def test_word_boundaries_prevent_substring_hits():
    """别名不能在单词内部命中——'the Americas' 不是 'America'。"""
    assert intel.find_location("dengue in the Americas is rising") is None
    assert intel.find_location("chinatown noodles") is None


# ---------- MOCK 模式：内置数据 ----------


@pytest.mark.parametrize(
    "written", ["Singapore", "新加坡", "Brazil", "巴西", "Thailand", "泰国", "泰國"]
)
def test_mock_canned_data_for_the_three_demo_countries(mock_mode, written):
    result = intel.lookup_dengue_context(written)
    assert result["matched"] is True
    assert result["endemicity"] == "high"
    assert result["season_note"]
    assert result["lookup_failed"] is False
    assert result["who_notices"]
    for notice in result["who_notices"]:
        assert set(notice) == {"title", "date", "url"}
        assert notice["url"].startswith(intel.WHO_ITEM_BASE)


def test_mock_unknown_location_is_not_matched(mock_mode):
    result = intel.lookup_dengue_context("Narnia")
    assert result["matched"] is False
    assert result["endemicity"] == "unknown"
    assert result["season_note"] is None
    assert result["lookup_failed"] is False


PAYLOAD_KEYS = {
    "location", "matched", "endemicity", "season_note", "who_notices", "lookup_failed",
}


def test_mock_payload_has_the_documented_shape(mock_mode):
    """MOCK 是环境，不是另一份契约。"""
    assert set(intel.lookup_dengue_context("Brazil")) == PAYLOAD_KEYS
    assert set(intel.lookup_dengue_context("Narnia")) == PAYLOAD_KEYS


def test_live_payload_has_the_same_shape(live_mode):
    assert set(intel.lookup_dengue_context("Brazil", fetcher=lambda: list(FAKE_ITEMS))) == PAYLOAD_KEYS
    intel.clear_notice_cache()
    assert set(intel.lookup_dengue_context("Narnia", fetcher=lambda: [])) == PAYLOAD_KEYS


# ---------- 通报筛选 ----------


def test_country_specific_notice_preferred(live_mode):
    result = intel.lookup_dengue_context("Brazil", fetcher=lambda: list(FAKE_ITEMS))
    assert [n["title"] for n in result["who_notices"]] == ["Dengue - Brazil"]
    assert result["who_notices"][0]["date"] == "2025-03-01"
    assert result["who_notices"][0]["url"].endswith("2025-DON900")


def test_falls_back_to_global_notices_newest_first(live_mode):
    result = intel.lookup_dengue_context("Singapore", fetcher=lambda: list(FAKE_ITEMS))
    titles = [n["title"] for n in result["who_notices"]]
    assert titles == ["Dengue - Global situation", "Dengue- Global situation"]
    # 退回全球通报是安全的：标题本身就写着这是全球形势，不会被当成新加坡的通报
    assert all("Global" in t for t in titles)


def test_notices_capped_at_three(live_mode):
    many = [
        {
            "Title": "Dengue - Global situation",
            "PublicationDateAndTime": f"202{i}-01-01T00:00:00Z",
            "UrlName": f"DON{i}",
        }
        for i in range(6)
    ]
    result = intel.lookup_dengue_context("Singapore", fetcher=lambda: many)
    assert len(result["who_notices"]) == intel.MAX_NOTICES == 3
    dates = [n["date"] for n in result["who_notices"]]
    assert dates == sorted(dates, reverse=True)


def test_items_without_urlname_are_dropped(live_mode):
    broken = [{"Title": "Dengue - Global situation", "PublicationDateAndTime": "2025-01-01T00:00:00Z"}]
    result = intel.lookup_dengue_context("Singapore", fetcher=lambda: broken)
    assert result["who_notices"] == []
    assert result["lookup_failed"] is False  # 拿到了列表，只是没有可用条目


# ---------- 缓存 ----------


def test_cache_prevents_a_second_fetch_within_12h(live_mode):
    calls = []

    def fetcher():
        calls.append(1)
        return list(FAKE_ITEMS)

    intel.lookup_dengue_context("Brazil", now=1000.0, fetcher=fetcher)
    intel.lookup_dengue_context("Thailand", now=1000.0 + intel.CACHE_TTL_SECONDS - 1, fetcher=fetcher)
    assert len(calls) == 1
    assert intel.notice_cache_state()["count"] == len(FAKE_ITEMS)


def test_cache_expires_after_12h(live_mode):
    calls = []

    def fetcher():
        calls.append(1)
        return list(FAKE_ITEMS)

    intel.lookup_dengue_context("Brazil", now=1000.0, fetcher=fetcher)
    intel.lookup_dengue_context("Brazil", now=1000.0 + intel.CACHE_TTL_SECONDS + 1, fetcher=fetcher)
    assert len(calls) == 2


def test_seeded_cache_is_used(live_mode):
    def boom():
        raise AssertionError("缓存有效期内不该再发请求")

    intel.seed_notice_cache(FAKE_ITEMS, fetched_at=5000.0)
    result = intel.lookup_dengue_context("Brazil", now=5001.0, fetcher=boom)
    assert result["who_notices"][0]["title"] == "Dengue - Brazil"


# ---------- 诚实的失败 ----------


def test_network_failure_reports_lookup_failed_and_no_notices(live_mode, monkeypatch):
    class BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            raise httpx.ConnectError("no network")

    monkeypatch.setattr(intel.httpx, "Client", BoomClient)
    result = intel.lookup_dengue_context("Brazil")

    assert result["lookup_failed"] is True
    assert result["who_notices"] == []
    # 地区表是本地的，网络挂了它照样能答
    assert result["matched"] is True
    assert result["endemicity"] == "high"
    assert result["season_note"]


def test_http_error_is_a_failure_not_a_guess(live_mode, monkeypatch):
    class ErrorClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            request = httpx.Request("GET", intel.WHO_DON_API)
            return httpx.Response(503, request=request)

    monkeypatch.setattr(intel.httpx, "Client", ErrorClient)
    result = intel.lookup_dengue_context("Singapore")
    assert result["lookup_failed"] is True
    assert result["who_notices"] == []


def test_malformed_payload_is_a_failure(live_mode):
    def bad_fetcher():
        raise intel.IntelLookupError("no value list")

    result = intel.lookup_dengue_context("Singapore", fetcher=bad_fetcher)
    assert result["lookup_failed"] is True
    assert result["who_notices"] == []


def test_failed_fetch_is_not_cached(live_mode):
    """失败不该被缓存成 12 小时的沉默：下一次还要再试。"""
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise intel.IntelLookupError("transient")
        return list(FAKE_ITEMS)

    first = intel.lookup_dengue_context("Brazil", now=1000.0, fetcher=flaky)
    second = intel.lookup_dengue_context("Brazil", now=1001.0, fetcher=flaky)
    assert first["lookup_failed"] is True
    assert second["lookup_failed"] is False
    assert len(attempts) == 2


# ---------- 工具契约 ----------


def test_tool_name_is_shared_by_every_layer():
    from app.prompt_builder import DENGUE_CONTEXT_TOOL

    assert intel.INTEL_TOOL_NAME == "lookup_dengue_context"
    assert DENGUE_CONTEXT_TOOL["function"]["name"] == intel.INTEL_TOOL_NAME
    assert DENGUE_CONTEXT_TOOL["function"]["parameters"]["required"] == ["location"]


def test_tool_description_tells_the_model_when_to_call_and_not_to_invent():
    from app.prompt_builder import DENGUE_CONTEXT_TOOL

    text = DENGUE_CONTEXT_TOOL["function"]["description"].lower()
    for phrase in ("travelling to", "living", "cite only the urls", "never construct", "plainly"):
        assert phrase in text
