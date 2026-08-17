"""联网检索的测试：协议解析、成本控制、/api/destination、追问的检索路径。

三条线索贯穿全文：

  **解析**：Anthropic 端点的返回是一串 content 块，真实链接埋在
  web_search_tool_result 里。用逐字段仿真的 fixture 覆盖四种块型、被截断的回复、
  以及「压根没检索」的返回。

  **花销**：检索按次计费，次数由模型决定。所以每条「不该检索」的路径都要有
  一个**计数假客户端**盯着——断言的是「调用次数为 0」，不是「结果看起来对」。

  **不许编造来源**：白名单现在是两层的并集（WHO 通报 + 检索结果）。
  白名单外的链接必须被拦下，两次都拦不住就退回兜底文案。

除了 RUN_LIVE_TESTS=1 时才跑的那一条，全部不发任何网络请求。
"""

import json
import logging
import os

import httpx
import pytest

from app.deepseek_client import (
    ANTHROPIC_MESSAGES_PATH,
    ANTHROPIC_VERSION,
    WEB_SEARCH_TOOL_TYPE,
    DeepSeekClient,
    DeepSeekError,
    parse_search_response,
    to_anthropic_messages,
)

WHO_PREFIX = "https://www.who.int/emergencies/disease-outbreak-news/item/"
NEA_URL = "https://www.nea.gov.sg/dengue-zika/dengue/dengue-cases"
MOH_URL = "https://www.moh.gov.sg/others/resources-and-statistics/infectious-disease-statistics"


# ---------- 仿真返回：逐字段照着真实响应写 ----------


def _full_payload(stop_reason: str = "end_turn") -> dict:
    """两轮检索、四种块型都出现的完整返回。"""
    return {
        "id": "msg_01ABCdef",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-v4-flash",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user is asking about Singapore. I should search for recent data.",
                "signature": "EqQBCkYIAxgCIkA...",
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {"query": "Singapore dengue cases last three months"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "Dengue Cases - National Environment Agency",
                        "url": NEA_URL,
                        "page_age": "2026-08-05",
                        "encrypted_content": "EvcBCioIAxgCIiQ...",
                    },
                    {
                        "type": "web_search_result",
                        "title": "Weekly Infectious Diseases Bulletin - MOH",
                        "url": MOH_URL,
                        "encrypted_content": "EvcBCioIAxgCIiR...",
                    },
                ],
            },
            {
                "type": "text",
                "text": "- Weekly dengue case counts have been falling since June 2026.",
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_02",
                "name": "web_search",
                "input": {"query": "Singapore NEA dengue clusters August 2026"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_02",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "Dengue Cases - National Environment Agency",
                        "url": NEA_URL,
                        "page_age": "2026-08-05",
                        "encrypted_content": "EvcBCioIAxgCIiQ...",
                    },
                    {
                        "type": "web_search_result",
                        "title": "Dengue - Global situation",
                        "url": WHO_PREFIX + "2024-DON518",
                        "page_age": "2024-05-30",
                        "encrypted_content": "EvcBCioIAxgCIiS...",
                    },
                ],
            },
            {"type": "text", "text": "- Active clusters remain concentrated in the east."},
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 13912,
            "output_tokens": 486,
            "server_tool_use": {"web_search_requests": 2},
        },
    }


NO_SEARCH_PAYLOAD = {
    "id": "msg_01NoSearch",
    "type": "message",
    "role": "assistant",
    "model": "deepseek-v4-flash",
    "content": [
        {"type": "thinking", "thinking": "No search needed for this one."},
        {"type": "text", "text": "I could not find recent public information about that place."},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 312, "output_tokens": 41},
}


# ---------- 假 httpx：断言我们真的按 Anthropic 协议发了请求 ----------


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom",
                request=httpx.Request("POST", "https://api.deepseek.com"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _fake_httpx(monkeypatch, payload, status_code: int = 200) -> list[dict]:
    """顶掉 httpx.AsyncClient，记录发出去的每个请求。返回捕获列表。"""
    captured: list[dict] = []

    class _FakeAsyncClient:
        def __init__(self, timeout=None) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            captured.append({"url": url, "json": json, "headers": headers})
            if isinstance(payload, httpx.HTTPError):
                raise payload
            return _FakeResponse(payload, status_code)

    monkeypatch.setattr("app.deepseek_client.httpx.AsyncClient", _FakeAsyncClient)
    return captured


class CountingSearch:
    """计数假客户端：只记「检索被调用了几次」，顺便返回一份可控结果。

    用**可调用实例**而不是函数顶掉 DeepSeekClient.chat_anthropic_search：
    实例不是描述符，因此不会被绑定成方法，签名里也就没有 self——
    调用参数就是流水线真正传进来的那些，断言起来一目了然。
    """

    def __init__(self, reply: str = "- Nothing much to report.", sources=None, search_count=1):
        self.calls: list[dict] = []
        self.reply = reply
        self.sources = list(sources or [])
        self.search_count = search_count

    async def __call__(self, system, messages, **kwargs):
        self.calls.append({"system": system, "messages": messages, **kwargs})
        return {
            "reply": self.reply,
            "sources": [dict(s) for s in self.sources],
            "search_count": self.search_count,
        }


# ---------- fixtures ----------


@pytest.fixture(autouse=True)
def _clean_caches():
    """目的地缓存与 WHO 通报缓存都是模块级的：每个测试都从空开始。

    不清的话，上一个测试塞进 WHO 缓存的空列表会在 12 小时内一直生效，
    下一个测试打的 fetch_who_notices 补丁根本不会被调用——排查起来很费时间。
    """
    from app.destination import clear_destination_cache
    from app.intel import clear_notice_cache

    clear_destination_cache()
    clear_notice_cache()
    yield
    clear_destination_cache()
    clear_notice_cache()


def _settings(monkeypatch, **env) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key.upper(), str(value))
    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture()
def client(monkeypatch):
    """MOCK_MODE=true：演示数据，永不出网。"""
    _settings(monkeypatch, mock_mode="true")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c

    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture()
def live_client(monkeypatch):
    """MOCK_MODE=false：走真实分支，但每个出网调用都在测试里被顶掉。"""
    _settings(monkeypatch, mock_mode="false")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c

    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def chat_body(**overrides) -> dict:
    body = {
        "language": "en",
        "question": "Should I be worried?",
        "context": {
            "dengue": {"score": 42.2, "level": "medium"},
            "worsening": {"score": 20.3, "level": "low"},
            "severe": {"score": 20.0, "level": "low"},
        },
        "history": [],
    }
    body.update(overrides)
    return body


# ================= 一、协议解析 =================


def test_parse_full_payload_keeps_text_and_harvests_every_url():
    outcome = parse_search_response(_full_payload())

    assert "Weekly dengue case counts" in outcome["reply"]
    assert "Active clusters" in outcome["reply"]
    # thinking 块绝不能进回复
    assert "No search needed" not in outcome["reply"]
    assert "I should search" not in outcome["reply"]

    urls = [s["url"] for s in outcome["sources"]]
    assert urls == [NEA_URL, MOH_URL, WHO_PREFIX + "2024-DON518"], "去重且保持出现顺序"
    assert outcome["sources"][0]["title"].startswith("Dengue Cases")
    assert outcome["sources"][0]["date"] == "2026-08-05"
    assert outcome["sources"][1]["date"] is None, "没有 page_age 就是 None，不许拿今天顶上"
    assert outcome["search_count"] == 2


def test_parse_warns_when_the_answer_was_truncated(caplog):
    with caplog.at_level(logging.WARNING, logger="app.deepseek_client"):
        outcome = parse_search_response(_full_payload(stop_reason="max_tokens"))

    assert "max_tokens" in caplog.text
    assert outcome["reply"], "截断的答案仍然要返回——半截答案比 502 有用"


def test_parse_payload_without_any_search_result():
    outcome = parse_search_response(NO_SEARCH_PAYLOAD)

    assert outcome["sources"] == []
    assert outcome["search_count"] == 0
    assert outcome["reply"].startswith("I could not find")


def test_parse_counts_server_tool_use_when_usage_is_missing():
    payload = _full_payload()
    payload.pop("usage")
    assert parse_search_response(payload)["search_count"] == 2


def test_parse_ignores_unknown_block_types():
    payload = _full_payload()
    payload["content"].append({"type": "some_future_block", "whatever": {"url": "x"}})
    outcome = parse_search_response(payload)
    assert len(outcome["sources"]) == 3, "未知块型不该被当成来源，也不该让解析炸掉"


def test_parse_only_takes_urls_from_search_result_blocks():
    """text 块里出现的链接不算来源——那是模型写的字，不是检索返回的东西。"""
    payload = {
        "content": [
            {"type": "text", "text": "See https://example.org/invented for details."},
        ],
        "stop_reason": "end_turn",
    }
    assert parse_search_response(payload)["sources"] == []


@pytest.mark.parametrize("payload", [None, "not a dict", {}, {"content": "nope"}])
def test_parse_rejects_malformed_payloads(payload):
    with pytest.raises(DeepSeekError):
        parse_search_response(payload)


def test_to_anthropic_messages_lifts_system_out_and_drops_blanks():
    converted = to_anthropic_messages(
        [
            {"role": "system", "content": "you are a helper"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": "hi"},
        ]
    )
    assert converted == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


# ================= 二、传输层 =================


@pytest.mark.anyio
async def test_search_call_uses_the_anthropic_endpoint_and_headers(monkeypatch):
    _settings(monkeypatch, mock_mode="false", deepseek_api_key="sk-test", deepseek_model="deepseek-v4-flash")
    captured = _fake_httpx(monkeypatch, _full_payload())

    outcome = await DeepSeekClient().chat_anthropic_search(
        "system text", [{"role": "user", "content": "Singapore?"}], language="en", max_uses=2
    )

    assert len(captured) == 1
    sent = captured[0]
    assert sent["url"].endswith(ANTHROPIC_MESSAGES_PATH)
    # OpenAI 那套鉴权头在这个端点上不管用
    assert sent["headers"]["x-api-key"] == "sk-test"
    assert sent["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    assert "Authorization" not in sent["headers"]
    body = sent["json"]
    assert body["system"] == "system text", "system 是顶层参数，不是一条消息"
    assert body["messages"] == [{"role": "user", "content": "Singapore?"}]
    assert body["tools"] == [
        {"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": 2}
    ]
    assert body["max_tokens"] >= 4000, "预算太小会被截断：实测 700 就不够"
    assert outcome["search_count"] == 2


@pytest.mark.anyio
async def test_no_search_tool_is_attached_when_max_uses_is_zero(monkeypatch):
    _settings(monkeypatch, mock_mode="false", deepseek_api_key="sk-test")
    captured = _fake_httpx(monkeypatch, NO_SEARCH_PAYLOAD)

    await DeepSeekClient().chat_anthropic_search(
        "s", [{"role": "user", "content": "q"}], language="en", max_uses=0
    )
    assert "tools" not in captured[0]["json"], "max_uses<=0 就是「这一轮不要检索」"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [httpx.ConnectError("no route"), _full_payload()],
    ids=["network-error", "http-500"],
)
async def test_transport_failures_become_deepseek_errors(monkeypatch, payload):
    _settings(monkeypatch, mock_mode="false", deepseek_api_key="sk-test")
    status = 200 if isinstance(payload, httpx.HTTPError) else 500
    _fake_httpx(monkeypatch, payload, status_code=status)

    with pytest.raises(DeepSeekError):
        await DeepSeekClient().chat_anthropic_search(
            "s", [{"role": "user", "content": "q"}], language="en"
        )


@pytest.mark.anyio
async def test_mock_mode_search_never_touches_the_network(monkeypatch):
    _settings(monkeypatch, mock_mode="true")
    captured = _fake_httpx(monkeypatch, _full_payload())

    outcome = await DeepSeekClient().chat_anthropic_search(
        "s", [{"role": "user", "content": "q"}], language="en", mock_location="Singapore"
    )

    assert captured == [], "演示模式一个请求都不该发"
    assert outcome["sources"], "演示模式也要给来源，才能走同一套引用校验"
    assert all(s["url"].startswith("http") for s in outcome["sources"])


@pytest.mark.anyio
async def test_mock_search_without_a_location_returns_no_sources(monkeypatch):
    _settings(monkeypatch, mock_mode="true")
    outcome = await DeepSeekClient().chat_anthropic_search(
        "s", [{"role": "user", "content": "q"}], language="en", mock_location=""
    )
    assert outcome["sources"] == []
    assert outcome["search_count"] == 0


# ================= 三、成本控制 =================


def test_a_question_without_a_place_buys_no_search(live_client, monkeypatch):
    """没有地点 = 不挂检索工具 = 一分钱都不花。用计数假客户端盯死这一条。"""
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch()
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)

    async def fake_tools(self, system, messages, tools, tool_executor, **kwargs):
        return {"reply": "The scores are relative indicators, and they are not probabilities.", "tool_results": []}

    monkeypatch.setattr(Client, "chat_with_tools", fake_tools)
    body = live_client.post("/api/chat", json=chat_body()).json()

    assert counter.calls == [], "没有地点就不该走检索路径"
    assert body["search_count"] == 0
    assert body["sources"] == []


def test_search_disabled_falls_back_to_the_tool_path(live_client, monkeypatch):
    _settings(monkeypatch, mock_mode="false", search_enabled="false")
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch()
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    used_tools = []

    async def fake_tools(self, system, messages, tools, tool_executor, **kwargs):
        used_tools.append(tools)
        return {"reply": "Singapore is worth protecting yourself in, and the tool found nothing.", "tool_results": []}

    monkeypatch.setattr(Client, "chat_with_tools", fake_tools)
    body = live_client.post(
        "/api/chat", json=chat_body(question="I am going to Singapore, is it risky?")
    ).json()

    assert counter.calls == [], "总开关关掉后，连有地点的问题也不许检索"
    assert used_tools, "但用户仍然要拿到回答——退回函数工具那条路"
    assert body["search_count"] == 0


def test_search_max_uses_setting_is_passed_through(live_client, monkeypatch):
    _settings(monkeypatch, mock_mode="false", search_max_uses="1")
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch(reply="- Cases in Brazil have been falling since June 2026.")
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    live_client.post("/api/chat", json=chat_body(question="What about Brazil right now?"))

    assert counter.calls[0]["max_uses"] == 1


def test_destination_cache_serves_the_second_request_for_free(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch(
        reply="- Cases have been falling since June 2026.",
        sources=[{"title": "NEA", "url": NEA_URL, "date": "2026-08-05"}],
    )
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    monkeypatch.setattr("app.intel.fetch_who_notices", lambda: [])

    first = live_client.post("/api/destination", json={"location": "Singapore", "language": "en"})
    second = live_client.post("/api/destination", json={"location": "singapore", "language": "en"})

    assert first.json()["search_status"] == "ok"
    assert len(counter.calls) == 1, "同一地点同一语言，第二次必须走缓存"
    assert second.json() == first.json()


def test_destination_cache_is_keyed_by_language(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch(
        reply="- Cases have been falling since June 2026.",
        sources=[{"title": "NEA", "url": NEA_URL, "date": None}],
    )
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    monkeypatch.setattr("app.intel.fetch_who_notices", lambda: [])

    live_client.post("/api/destination", json={"location": "Singapore", "language": "en"})
    live_client.post("/api/destination", json={"location": "Singapore", "language": "zh-CN"})
    assert len(counter.calls) == 2, "换一种语言就是另一份回答，不能共用缓存"


@pytest.mark.anyio
async def test_destination_cache_expires_after_the_ttl(monkeypatch):
    """注入时钟，把时间推过 TTL：缓存必须失效并重新检索。"""
    _settings(monkeypatch, mock_mode="true", search_cache_ttl_seconds="60")
    from app.destination import run_destination
    from app.schemas import DestinationRequest

    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch(
        reply="- Cases have been falling since June 2026.",
        sources=[{"title": "NEA", "url": NEA_URL, "date": None}],
    )
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)

    req = DestinationRequest(location="Singapore", language="en")
    await run_destination(req, now=1000.0)
    await run_destination(req, now=1030.0)
    assert len(counter.calls) == 1, "TTL 之内不该再查"

    await run_destination(req, now=1000.0 + 61)
    assert len(counter.calls) == 2, "TTL 之后必须重新查"


def test_a_failed_lookup_is_not_cached(live_client, monkeypatch):
    """一次网络抖动不能被钉在缓存里 6 小时。"""
    from app.deepseek_client import DeepSeekClient as Client

    calls = []

    async def boom(self, system, messages, **kwargs):
        calls.append(kwargs)
        raise DeepSeekError("upstream down")

    monkeypatch.setattr(Client, "chat_anthropic_search", boom)
    monkeypatch.setattr("app.intel.fetch_who_notices", lambda: [])

    first = live_client.post("/api/destination", json={"location": "Brazil", "language": "en"})
    second = live_client.post("/api/destination", json={"location": "Brazil", "language": "en"})

    assert first.json()["search_status"] == "degraded"
    assert len(calls) == 2, "失败的那次不该进缓存"
    assert second.status_code == 200


# ================= 四、/api/destination =================


def test_destination_matched_location_has_three_layers(client):
    body = client.post("/api/destination", json={"location": "Singapore", "language": "en"}).json()

    assert body["matched"] is True
    assert body["location"] == "Singapore"
    assert body["endemicity"] == "high"
    assert body["season_note"]
    assert body["who_notices"], "第 1 层：WHO 通报"
    assert body["recent_findings"], "第 2 层：检索要点"
    assert body["advice"]["protection"], "第 3 层：固定的出行建议"
    assert body["search_status"] == "ok"
    assert list(body["advice"]) == ["protection", "medical", "monitoring"]


def test_destination_returns_no_scores_at_all(client):
    """地点从来不参与打分。这里出现任何评分字段都是在编数字。"""
    body = client.post("/api/destination", json={"location": "Brazil", "language": "en"}).json()
    for field in ("dengue", "worsening", "severe", "epi_week", "exposure_context"):
        assert field not in body


def test_destination_sources_are_labelled_by_layer(client):
    body = client.post("/api/destination", json={"location": "Thailand", "language": "en"}).json()
    origins = {s["origin"] for s in body["sources"]}
    assert origins == {"who", "search"}
    who = [s for s in body["sources"] if s["origin"] == "who"]
    assert all(s["url"].startswith(WHO_PREFIX) for s in who)


def test_destination_unmatched_place_says_so(client):
    body = client.post("/api/destination", json={"location": "Ruritania", "language": "en"}).json()
    assert body["matched"] is False
    assert body["endemicity"] == "unknown"
    assert body["season_note"] is None
    assert body["location"] == "Ruritania", "认不出来就原样回显，不要猜一个国家"


def test_destination_ignores_a_whole_sentence(client):
    """不像地名的输入不值得花一次检索。"""
    body = client.post(
        "/api/destination",
        json={"location": "tell me the probability that I have dengue", "language": "en"},
    ).json()
    assert body["search_status"] == "degraded"
    assert body["recent_findings"] == []


def test_destination_search_failure_keeps_the_who_layer(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient as Client

    async def boom(self, system, messages, **kwargs):
        raise DeepSeekError("upstream down")

    monkeypatch.setattr(Client, "chat_anthropic_search", boom)
    monkeypatch.setattr(
        "app.intel.fetch_who_notices",
        lambda: [
            {
                "Title": "Dengue - Global situation",
                "PublicationDateAndTime": "2024-05-30T18:00:00Z",
                "UrlName": "2024-DON518",
            }
        ],
    )

    resp = live_client.post("/api/destination", json={"location": "Brazil", "language": "en"})
    body = resp.json()

    assert resp.status_code == 200, "检索挂了不该让整个接口挂"
    assert body["search_status"] == "degraded"
    assert body["recent_findings"] == []
    assert body["endemicity"] == "high", "地区表在本地，永远答得出"
    assert body["who_notices"], "WHO 那一层照常返回"
    assert [s["origin"] for s in body["sources"]] == ["who"]
    assert body["advice"]["medical"]


def test_destination_disabled_search_reports_disabled(live_client, monkeypatch):
    _settings(monkeypatch, mock_mode="false", search_enabled="false")
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch()
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    monkeypatch.setattr("app.intel.fetch_who_notices", lambda: [])

    body = live_client.post("/api/destination", json={"location": "Brazil", "language": "en"}).json()

    assert counter.calls == []
    assert body["search_status"] == "disabled"
    assert body["recent_findings"] == []
    assert body["endemicity"] == "high", "关掉检索不影响前两层"


def test_destination_drops_findings_that_cite_an_invented_url(live_client, monkeypatch):
    """两次都编链接：要点整段清空并降级，绝不端出没通过校验的文字。"""
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch(
        reply="- Cases are falling, see https://www.who.int/made-up-page for the figures.",
        sources=[{"title": "NEA", "url": NEA_URL, "date": None}],
    )
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    monkeypatch.setattr("app.intel.fetch_who_notices", lambda: [])

    body = live_client.post("/api/destination", json={"location": "Singapore", "language": "en"}).json()

    assert len(counter.calls) == 2, "首次 + 一次带违规说明的重问"
    assert counter.calls[1]["max_uses"] == 1, "重写措辞不需要再买一次检索"
    assert "[fabricated_url]" in counter.calls[1]["messages"][-1]["content"]
    assert body["search_status"] == "degraded"
    assert body["recent_findings"] == []
    assert "made-up-page" not in json.dumps(body)


def test_destination_accepts_findings_grounded_in_the_who_layer(live_client, monkeypatch):
    """反方向：引用的是 WHO 那层给过的链接，就该放行——白名单是两层的并集。"""
    from app.deepseek_client import DeepSeekClient as Client

    who_url = WHO_PREFIX + "2024-DON518"
    counter = CountingSearch(
        reply=f"- WHO's global update ({who_url}) still lists the country as affected.",
        sources=[],
        search_count=1,
    )
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    monkeypatch.setattr(
        "app.intel.fetch_who_notices",
        lambda: [
            {
                "Title": "Dengue - Global situation",
                "PublicationDateAndTime": "2024-05-30T18:00:00Z",
                "UrlName": "2024-DON518",
            }
        ],
    )

    body = live_client.post("/api/destination", json={"location": "Brazil", "language": "en"}).json()

    assert len(counter.calls) == 1, "没有违规就不该重问"
    assert body["search_status"] == "ok"
    assert body["recent_findings"] and who_url in body["recent_findings"][0]


@pytest.mark.parametrize("language", ["zh-CN", "zh-TW", "en", "es", "pt"])
def test_destination_is_localised_and_clean_in_every_language(client, language):
    """出行建议是兜底文案：它必须在五种语言下都干净，否则安全网本身不安全。"""
    from app.verifier import verify_chat_reply

    body = client.post("/api/destination", json={"location": "Brazil", "language": language}).json()

    assert body["disclaimer"] and body["model_note"]
    texts = [t for items in body["advice"].values() for t in items] + body["recent_findings"]
    allowed = [s["url"] for s in body["sources"]]
    assert verify_chat_reply("\n".join(texts), language, allowed) == []


@pytest.mark.parametrize(
    "location", ["", "   ", "x" * 121]
)
def test_destination_rejects_bad_locations(client, location):
    assert client.post("/api/destination", json={"location": location}).status_code == 422


def test_destination_endemic_and_quiet_advice_differ(client):
    endemic = client.post("/api/destination", json={"location": "Brazil", "language": "en"}).json()
    quiet = client.post("/api/destination", json={"location": "Iceland", "language": "en"}).json()
    assert endemic["advice"]["medical"] != quiet["advice"]["medical"]
    assert endemic["advice"]["protection"] == quiet["advice"]["protection"]


def test_parse_findings_drops_bullets_and_link_only_lines():
    from app.destination import parse_findings

    reply = (
        "Here is what I found:\n"
        "- Cases fell through July.\n"
        "* Clusters remain in the east.\n"
        "https://example.org/only-a-link\n"
        "- " + "x" * 500 + "\n"
    )
    assert parse_findings(reply) == ["Cases fell through July.", "Clusters remain in the east."]


def test_parse_findings_falls_back_to_plain_lines():
    from app.destination import parse_findings

    assert parse_findings("No recent information was found.") == [
        "No recent information was found."
    ]


def test_parse_findings_strips_markdown_the_model_added_anyway():
    """实测：即便提示词说了别用 Markdown，模型还是会写 **加粗**。"""
    from app.destination import parse_findings

    assert parse_findings("- **Cases remain low:** 1,180 in H1 2026.") == [
        "Cases remain low: 1,180 in H1 2026."
    ]


def test_search_sources_are_capped_but_never_drop_a_cited_one():
    """实测一次请求带回 20 条结果。截断可以，但截掉被引用的那条会自伤。"""
    from app.schemas import MAX_SEARCH_SOURCES, select_search_sources

    # 编号补零：否则 ".../1" 是 ".../17" 的前缀，子串匹配会把两条都算成被引用
    sources = [{"title": f"r{i}", "url": f"https://example.org/p{i:02d}"} for i in range(20)]
    reply = "See https://example.org/p17 for the weekly figures."
    chosen = select_search_sources(sources, reply)

    urls = [s["url"] for s in chosen]
    assert len(urls) == MAX_SEARCH_SOURCES
    assert urls[0] == "https://example.org/p17", "被引用的排最前，绝不能被截掉"
    assert urls[1:] == [f"https://example.org/p{i:02d}" for i in range(MAX_SEARCH_SOURCES - 1)]


def test_every_cited_source_survives_even_past_the_cap():
    from app.schemas import select_search_sources

    sources = [{"title": f"r{i}", "url": f"https://example.org/p{i:02d}"} for i in range(20)]
    reply = " ".join(f"https://example.org/p{i:02d}" for i in range(12))
    chosen = select_search_sources(sources, reply)
    assert len(chosen) == 12, "引用了 12 条就得留 12 条，白名单不能比回复窄"


# ================= 五、追问对话的检索路径 =================


def test_chat_with_a_place_uses_search_and_labels_both_layers(client):
    body = client.post(
        "/api/chat", json=chat_body(question="I am flying to Singapore next month.")
    ).json()

    assert body["search_count"] >= 1
    assert {s["origin"] for s in body["sources"]} == {"who", "search"}


def test_chat_search_whitelist_is_the_union_of_both_layers(live_client, monkeypatch):
    """引用 WHO 工具给的链接 + 检索给的链接，两个都必须放行。"""
    from app.deepseek_client import DeepSeekClient as Client

    who_url = WHO_PREFIX + "2024-DON518"
    counter = CountingSearch(
        reply=f"Brazil is highly endemic ({who_url}); the latest counts are at {NEA_URL}.",
        sources=[{"title": "NEA", "url": NEA_URL, "date": None}],
    )
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    monkeypatch.setattr(
        "app.intel.fetch_who_notices",
        lambda: [
            {
                "Title": "Dengue - Global situation",
                "PublicationDateAndTime": "2024-05-30T18:00:00Z",
                "UrlName": "2024-DON518",
            }
        ],
    )

    body = live_client.post(
        "/api/chat", json=chat_body(question="What is happening in Brazil?")
    ).json()

    assert len(counter.calls) == 1
    urls = {(s["url"], s["origin"]) for s in body["sources"]}
    assert urls == {(who_url, "who"), (NEA_URL, "search")}
    assert body["search_count"] == 1


def test_chat_search_reply_citing_an_off_whitelist_url_is_rejected(live_client, monkeypatch):
    """检索路径上的 fabricated_url：两次都编，退回兜底句并清空 sources。"""
    from app.deepseek_client import DeepSeekClient as Client
    from app.pipeline import _UNRELIABLE_REPLY

    counter = CountingSearch(
        reply="Brazil is fine — see https://www.who.int/invented-dengue-page for the numbers.",
        sources=[{"title": "NEA", "url": NEA_URL, "date": None}],
    )
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    monkeypatch.setattr("app.intel.fetch_who_notices", lambda: [])

    resp = live_client.post("/api/chat", json=chat_body(question="Is Brazil risky right now?"))
    body = resp.json()

    assert resp.status_code == 200
    assert len(counter.calls) == 2
    assert counter.calls[1]["max_uses"] == 0, "重写措辞不该再买一次检索"
    assert body["reply"] == _UNRELIABLE_REPLY["en"]
    assert body["sources"] == []
    assert "invented-dengue-page" not in body["reply"]


def test_chat_search_upstream_failure_is_a_502(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient as Client

    async def boom(self, system, messages, **kwargs):
        raise DeepSeekError("upstream down")

    monkeypatch.setattr(Client, "chat_anthropic_search", boom)
    resp = live_client.post("/api/chat", json=chat_body(question="How is Thailand doing?"))
    assert resp.status_code == 502


def test_chat_search_prompt_carries_the_intel_baseline(live_client, monkeypatch):
    """检索这一轮不给函数工具，地区背景直接摆进提示词——包括可引用的 WHO 链接。"""
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch(reply="Thailand has a rainy season, and you should use repellent.")
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    monkeypatch.setattr(
        "app.intel.fetch_who_notices",
        lambda: [
            {
                "Title": "Dengue - Global situation",
                "PublicationDateAndTime": "2024-05-30T18:00:00Z",
                "UrlName": "2024-DON518",
            }
        ],
    )

    live_client.post("/api/chat", json=chat_body(question="Going to Thailand in July."))

    user_text = counter.calls[0]["messages"][0]["content"]
    assert "Thailand" in user_text
    assert WHO_PREFIX + "2024-DON518" in user_text
    assert "S1." in counter.calls[0]["system"], "检索纪律条款必须在 system 里"


# ================= 六、花销回流与统计 =================


def test_search_spend_is_logged_for_every_request_that_could_search(client, tmp_path, monkeypatch):
    log = tmp_path / "spend.jsonl"
    monkeypatch.setenv("EVAL_LOG_PATH", str(log))
    from app.config import get_settings

    get_settings.cache_clear()

    client.post("/api/destination", json={"location": "Singapore", "language": "en"})
    client.post("/api/chat", json=chat_body(question="I am going to Brazil."))

    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    kinds = {r["kind"]: r for r in records if "search_count" in r}
    assert set(kinds) == {"destination", "chat"}
    assert kinds["destination"]["location"] == "Singapore"
    assert kinds["chat"]["search_count"] >= 1
    # 问题原文绝不落盘
    assert "I am going to Brazil" not in log.read_text(encoding="utf-8")


def test_eval_stats_reports_search_spend():
    from scripts.eval_stats import compute_stats

    records = [
        {"kind": "chat", "search_count": 0, "search_status": "ok", "language": "en"},
        {"kind": "chat", "search_count": 4, "search_status": "ok", "language": "en"},
        {"kind": "destination", "search_count": 2, "search_status": "degraded", "language": "en"},
    ]
    stats = compute_stats(records)

    assert stats["total"] == 0, "检索记录不是评估记录，不该混进模型统计"
    search = stats["search"]
    assert search["n"] == 3
    assert search["total"] == 6
    assert search["mean"] == 2.0
    assert search["max"] == 4
    assert search["zero"] == 1
    assert search["by_kind"]["chat"] == {"n": 2, "total": 4, "mean": 2.0}
    assert search["statuses"] == {"degraded": 1, "ok": 2}


def test_eval_stats_keeps_both_record_kinds(tmp_path):
    from scripts.eval_stats import load_records

    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"scores": {"dengue": {"score": 10.0, "level": "low"}}}),
                json.dumps({"kind": "chat", "search_count": 2}),
                json.dumps({"neither": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records, skipped = load_records(path)
    assert len(records) == 2
    assert skipped == 1


# ================= 七、真调用一次（默认跳过，CI 永远不花钱） =================


@pytest.mark.anyio
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="设置 RUN_LIVE_TESTS=1 才会真的调用 DeepSeek 并产生费用",
)
async def test_live_search_returns_at_least_one_real_source(monkeypatch):
    """唯一一条真的出网的测试。默认跳过——CI 不该为一次冒烟买单。

    只断言最低限度的东西：真的检索了、真的带回了 http 开头的链接。
    具体内容会随时间变化，断言它就是在断言今天的新闻。
    """
    _settings(monkeypatch, mock_mode="false")
    from app.config import get_settings

    settings = get_settings()
    if not settings.deepseek_api_key:
        pytest.skip("没有配置 DEEPSEEK_API_KEY")

    outcome = await DeepSeekClient().chat_anthropic_search(
        "You are a public-health information assistant. Answer in English, in two short bullets.",
        [
            {
                "role": "user",
                "content": (
                    "Search the web: what is the dengue situation in Singapore over the last "
                    "three months? Give two short bullets and prefer official sources."
                ),
            }
        ],
        language="en",
        max_uses=1,
    )

    assert outcome["search_count"] >= 1, "这一轮应当真的检索过"
    assert outcome["sources"], "检索应当带回至少一条来源"
    assert all(s["url"].startswith("http") for s in outcome["sources"])
    assert outcome["reply"].strip()


# ---------- 来源权威性标注 ----------


class TestSourceAuthority:
    """检索结果里卫生部门与新闻聚合站并排出现，读者有权一眼分清。

    判定只看域名，因此是可核对的事实，不是对报道质量的评价。
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.nea.gov.sg/dengue-zika/dengue",   # 新加坡环境局
            "https://doh.gov.ph/advisory",                 # 菲律宾卫生部
            "https://ddc.moph.go.th/report",               # 泰国卫生部
            "https://www.gob.mx/salud",                    # 墨西哥卫生部
            "https://www.gouv.fr/sante",                   # 法国政府
            "https://www.govt.nz/travel",                  # 新西兰政府
            "https://www.cdc.gov/dengue",                  # 顶级域即 .gov
            "https://www.who.int/emergencies/x",           # .int
            "https://www.ecdc.europa.eu/en/dengue-monthly",  # 允许清单里的父域
            "https://www.paho.org/en/dengue",
        ],
    )
    def test_official_domains(self, url):
        from app.schemas import classify_authority

        assert classify_authority(url) == "official"

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.malaymail.com/news/x",
            "http://www.china.org.cn/world/x.shtml",
            "https://www.magzter.com/stories/x",
            "https://denguevisualatlas.com/th",
            "https://go.com/whatever",          # 陷阱：go 不跟两字母国家码
            "https://gov.example.com/fake",     # 陷阱：gov 在最前面不算
            "https://notgov.sg/x",
            "",
            "not a url",
        ],
    )
    def test_non_official_domains(self, url):
        from app.schemas import classify_authority

        assert classify_authority(url) == "other"

    def test_who_notices_are_always_official(self):
        from app.schemas import merge_sources

        merged = merge_sources(
            [{"title": "Dengue - Global", "date": "2024-05-30", "url": "https://www.who.int/x"}],
            [],
        )
        assert merged[0].origin == "who"
        assert merged[0].authority == "official"

    def test_official_search_results_sort_first_without_dropping_any(self):
        """只调顺序，不丢条目——sources 同时是校验器白名单。"""
        from app.schemas import merge_sources

        merged = merge_sources(
            [],
            [
                {"title": "News", "url": "https://www.malaymail.com/a"},
                {"title": "NEA", "url": "https://www.nea.gov.sg/b"},
                {"title": "Aggregator", "url": "https://www.magzter.com/c"},
                {"title": "MOH", "url": "https://ddc.moph.go.th/d"},
            ],
        )
        assert [s.authority for s in merged] == ["official", "official", "other", "other"]
        # 一条都不能少，否则模型真正引用过的链接可能被自己的校验器判成编造
        assert len(merged) == 4
        # 同类之间保持检索返回的原顺序
        assert [s.title for s in merged] == ["NEA", "MOH", "News", "Aggregator"]
