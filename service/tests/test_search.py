"""Web search tests: protocol parsing, cost control, /api/destination, the chat path.

Three threads run through the whole file:

  **Parsing**: the Anthropic endpoint returns a run of content blocks, with the real
  links buried inside web_search_tool_result. Field-by-field simulated fixtures cover
  the four block types, a truncated reply, and a "did not search at all" response.

  **Spend**: search is billed per call and the model decides how many calls to make.
  So every path that must not search has a **counting fake client** watching it -- the
  assertion is "the call count is 0", not "the result looks right".

  **No invented sources**: the allow-list is now the union of two layers (WHO notices
  + search results). Links outside it must be blocked, and if both attempts fail the
  fallback copy is served instead.

Apart from the one test that runs only with RUN_LIVE_TESTS=1, nothing here goes online.
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


# ---------- Simulated responses: written field by field from real ones ----------


def _full_payload(stop_reason: str = "end_turn") -> dict:
    """A complete response with two search rounds and all four block types present."""
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


# ---------- Fake httpx: assert we really send the Anthropic protocol ----------


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
    """Stub out httpx.AsyncClient and record every request sent. Returns the captures."""
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
    """Counting fake client: records how often search was called, returns a canned result.

    DeepSeekClient.chat_anthropic_search is replaced by a **callable instance** rather
    than a function: an instance is not a descriptor, so it is never bound as a method
    and there is no self in the signature -- the call arguments are exactly what the
    pipeline really passed in, which makes the assertions obvious.
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
    """The destination cache and the WHO notice cache are module level: start each empty.

    Without clearing, an empty list one test pushed into the WHO cache stays in effect
    for 12 hours, and the next test's fetch_who_notices patch is never called at all --
    which costs a lot of time to track down.
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
    """MOCK_MODE=true: demo data, never goes online."""
    _settings(monkeypatch, mock_mode="true")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c

    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture()
def live_client(monkeypatch):
    """MOCK_MODE=false: takes the real branch, but every outbound call is stubbed here."""
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


# ================= 1. Protocol parsing =================


def test_parse_full_payload_keeps_text_and_harvests_every_url():
    outcome = parse_search_response(_full_payload())

    assert "Weekly dengue case counts" in outcome["reply"]
    assert "Active clusters" in outcome["reply"]
    # thinking blocks must never reach the reply
    assert "No search needed" not in outcome["reply"]
    assert "I should search" not in outcome["reply"]

    urls = [s["url"] for s in outcome["sources"]]
    assert urls == [NEA_URL, MOH_URL, WHO_PREFIX + "2024-DON518"], "deduplicated, first-seen order"
    assert outcome["sources"][0]["title"].startswith("Dengue Cases")
    assert outcome["sources"][0]["date"] == "2026-08-05"
    assert outcome["sources"][1]["date"] is None, "no page_age means None, not today's date"
    assert outcome["search_count"] == 2


def test_parse_warns_when_the_answer_was_truncated(caplog):
    with caplog.at_level(logging.WARNING, logger="app.deepseek_client"):
        outcome = parse_search_response(_full_payload(stop_reason="max_tokens"))

    assert "max_tokens" in caplog.text
    assert outcome["reply"], "a truncated answer is still returned -- half beats a 502"


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
    assert len(outcome["sources"]) == 3, "unknown block types: not sources, must not crash parsing"


def test_parse_only_takes_urls_from_search_result_blocks():
    """A link in a text block is not a source -- the model wrote it, search did not."""
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


# ================= 2. Transport layer =================


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
    # The OpenAI-style auth header does not work on this endpoint
    assert sent["headers"]["x-api-key"] == "sk-test"
    assert sent["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    assert "Authorization" not in sent["headers"]
    body = sent["json"]
    assert body["system"] == "system text", "system is a top-level parameter, not a message"
    assert body["messages"] == [{"role": "user", "content": "Singapore?"}]
    assert body["tools"] == [
        {"type": WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": 2}
    ]
    assert body["max_tokens"] >= 4000, "too small a budget truncates: 700 measured as not enough"
    assert outcome["search_count"] == 2


@pytest.mark.anyio
async def test_no_search_tool_is_attached_when_max_uses_is_zero(monkeypatch):
    _settings(monkeypatch, mock_mode="false", deepseek_api_key="sk-test")
    captured = _fake_httpx(monkeypatch, NO_SEARCH_PAYLOAD)

    await DeepSeekClient().chat_anthropic_search(
        "s", [{"role": "user", "content": "q"}], language="en", max_uses=0
    )
    assert "tools" not in captured[0]["json"], "max_uses<=0 means 'no search this round'"


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

    assert captured == [], "demo mode must not send a single request"
    assert outcome["sources"], "demo mode returns sources too, so the same citation check runs"
    assert all(s["url"].startswith("http") for s in outcome["sources"])


@pytest.mark.anyio
async def test_mock_search_without_a_location_returns_no_sources(monkeypatch):
    _settings(monkeypatch, mock_mode="true")
    outcome = await DeepSeekClient().chat_anthropic_search(
        "s", [{"role": "user", "content": "q"}], language="en", mock_location=""
    )
    assert outcome["sources"] == []
    assert outcome["search_count"] == 0


# ================= 3. Cost control =================


def test_a_question_without_a_place_buys_no_search(live_client, monkeypatch):
    """No place = no search tool attached = no spend. The counting fake client pins it."""
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch()
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)

    async def fake_tools(self, system, messages, tools, tool_executor, **kwargs):
        return {"reply": "The scores are relative indicators, and they are not probabilities.", "tool_results": []}

    monkeypatch.setattr(Client, "chat_with_tools", fake_tools)
    body = live_client.post("/api/chat", json=chat_body()).json()

    assert counter.calls == [], "no place, so the search path must not be taken"
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

    assert counter.calls == [], "master switch off: not even a question with a place may search"
    assert used_tools, "the user still gets an answer -- it falls back to the function tools"
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
    assert len(counter.calls) == 1, "same place and language: the second call must hit the cache"
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
    assert len(counter.calls) == 2, "another language is another answer; no shared cache"


@pytest.mark.anyio
async def test_destination_cache_expires_after_the_ttl(monkeypatch):
    """Inject the clock and push time past the TTL: the cache expires and re-searches."""
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
    assert len(counter.calls) == 1, "no second lookup within the TTL"

    await run_destination(req, now=1000.0 + 61)
    assert len(counter.calls) == 2, "a fresh lookup is required after the TTL"


def test_a_failed_lookup_is_not_cached(live_client, monkeypatch):
    """One network blip must not be pinned in the cache for 6 hours."""
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
    assert len(calls) == 2, "the failed attempt must not be cached"
    assert second.status_code == 200


# ================= 4. /api/destination =================


def test_destination_matched_location_has_three_layers(client):
    body = client.post("/api/destination", json={"location": "Singapore", "language": "en"}).json()

    assert body["matched"] is True
    assert body["location"] == "Singapore"
    assert body["endemicity"] == "high"
    assert body["season_note"]
    assert body["who_notices"], "layer 1: WHO notices"
    assert body["recent_findings"], "layer 2: search findings"
    assert body["advice"]["protection"], "layer 3: the fixed travel advice"
    assert body["search_status"] == "ok"
    assert list(body["advice"]) == ["protection", "medical", "monitoring"]


def test_destination_returns_no_scores_at_all(client):
    """A place never takes part in scoring. Any score field here would be invented."""
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
    assert body["location"] == "Ruritania", "echo it back unchanged, do not guess a country"


def test_destination_ignores_a_whole_sentence(client):
    """Input that does not look like a place name is not worth buying a search for."""
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

    assert resp.status_code == 200, "a failed search must not take the endpoint down"
    assert body["search_status"] == "degraded"
    assert body["recent_findings"] == []
    assert body["endemicity"] == "high", "the region table is local and always answers"
    assert body["who_notices"], "the WHO layer is returned as usual"
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
    assert body["endemicity"] == "high", "search off does not affect the first two layers"


def test_destination_drops_findings_that_cite_an_invented_url(live_client, monkeypatch):
    """Invented links twice: findings are cleared and degraded, never served unchecked."""
    from app.deepseek_client import DeepSeekClient as Client

    counter = CountingSearch(
        reply="- Cases are falling, see https://www.who.int/made-up-page for the figures.",
        sources=[{"title": "NEA", "url": NEA_URL, "date": None}],
    )
    monkeypatch.setattr(Client, "chat_anthropic_search", counter)
    monkeypatch.setattr("app.intel.fetch_who_notices", lambda: [])

    body = live_client.post("/api/destination", json={"location": "Singapore", "language": "en"}).json()

    assert len(counter.calls) == 2, "the first call plus one re-ask with the violations"
    assert counter.calls[1]["max_uses"] == 1, "rewording needs no second paid search"
    assert "[fabricated_url]" in counter.calls[1]["messages"][-1]["content"]
    assert body["search_status"] == "degraded"
    assert body["recent_findings"] == []
    assert "made-up-page" not in json.dumps(body)


def test_destination_accepts_findings_grounded_in_the_who_layer(live_client, monkeypatch):
    """Other direction: a link the WHO layer already gave passes -- the list is a union."""
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

    assert len(counter.calls) == 1, "no violation, no re-ask"
    assert body["search_status"] == "ok"
    assert body["recent_findings"] and who_url in body["recent_findings"][0]


@pytest.mark.parametrize("language", ["zh-CN", "zh-TW", "en", "es", "pt"])
def test_destination_is_localised_and_clean_in_every_language(client, language):
    """Travel advice is fallback copy: clean in five languages, or the safety net is not."""
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
    """Measured: even though the prompt says no Markdown, the model still writes **bold**."""
    from app.destination import parse_findings

    assert parse_findings("- **Cases remain low:** 1,180 in H1 2026.") == [
        "Cases remain low: 1,180 in H1 2026."
    ]


def test_search_sources_are_capped_but_never_drop_a_cited_one():
    """Measured at 20 results per request. Capping is fine; cutting a cited one backfires."""
    from app.schemas import MAX_SEARCH_SOURCES, select_search_sources

    # Zero-padded numbering: otherwise ".../1" prefixes ".../17" and substring matching
    # would count both as cited
    sources = [{"title": f"r{i}", "url": f"https://example.org/p{i:02d}"} for i in range(20)]
    reply = "See https://example.org/p17 for the weekly figures."
    chosen = select_search_sources(sources, reply)

    urls = [s["url"] for s in chosen]
    assert len(urls) == MAX_SEARCH_SOURCES
    assert urls[0] == "https://example.org/p17", "the cited one sorts first, never cut"
    assert urls[1:] == [f"https://example.org/p{i:02d}" for i in range(MAX_SEARCH_SOURCES - 1)]


def test_every_cited_source_survives_even_past_the_cap():
    from app.schemas import select_search_sources

    sources = [{"title": f"r{i}", "url": f"https://example.org/p{i:02d}"} for i in range(20)]
    reply = " ".join(f"https://example.org/p{i:02d}" for i in range(12))
    chosen = select_search_sources(sources, reply)
    assert len(chosen) == 12, "cite 12, keep 12: allow-list must not be narrower than the reply"


# ================= 5. The chat search path =================


def test_chat_with_a_place_uses_search_and_labels_both_layers(client):
    body = client.post(
        "/api/chat", json=chat_body(question="I am flying to Singapore next month.")
    ).json()

    assert body["search_count"] >= 1
    assert {s["origin"] for s in body["sources"]} == {"who", "search"}


def test_chat_search_whitelist_is_the_union_of_both_layers(live_client, monkeypatch):
    """A link from the WHO tool plus a link from search: both must be allowed."""
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
    """fabricated_url on the search path: invented twice, fall back and clear sources."""
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
    assert counter.calls[1]["max_uses"] == 0, "rewording must not buy another search"
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
    """No tools this round; region background goes into the prompt, citable WHO links too."""
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
    assert "S1." in counter.calls[0]["system"], "the search discipline clauses live in system"


# ================= 6. Spend logging and statistics =================


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
    # The raw question is never written to disk
    assert "I am going to Brazil" not in log.read_text(encoding="utf-8")


def test_eval_stats_reports_search_spend():
    from scripts.eval_stats import compute_stats

    records = [
        {"kind": "chat", "search_count": 0, "search_status": "ok", "language": "en"},
        {"kind": "chat", "search_count": 4, "search_status": "ok", "language": "en"},
        {"kind": "destination", "search_count": 2, "search_status": "degraded", "language": "en"},
    ]
    stats = compute_stats(records)

    assert stats["total"] == 0, "search records are not eval records, not model stats"
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


# ================= 7. One real call (skipped by default, CI never pays) =================


@pytest.mark.anyio
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to really call DeepSeek and incur a cost",
)
async def test_live_search_returns_at_least_one_real_source(monkeypatch):
    """The only test that really goes online, skipped by default -- CI must not pay for it.

    It asserts the bare minimum: a search really happened, and links starting with http
    really came back. The exact content changes over time, so asserting it would mean
    asserting today's news.
    """
    _settings(monkeypatch, mock_mode="false")
    from app.config import get_settings

    settings = get_settings()
    if not settings.deepseek_api_key:
        pytest.skip("DEEPSEEK_API_KEY is not configured")

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

    assert outcome["search_count"] >= 1, "this round should really have searched"
    assert outcome["sources"], "the search should return at least one source"
    assert all(s["url"].startswith("http") for s in outcome["sources"])
    assert outcome["reply"].strip()


# ---------- Source authority labelling ----------


class TestSourceAuthority:
    """Health bodies and news aggregators sit side by side; readers must tell them apart.

    The call is made on the domain alone: a verifiable fact, not a verdict on reporting.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.nea.gov.sg/dengue-zika/dengue",   # Singapore environment agency
            "https://doh.gov.ph/advisory",                 # Philippines health department
            "https://ddc.moph.go.th/report",               # Thailand health ministry
            "https://www.gob.mx/salud",                    # Mexico health ministry
            "https://www.gouv.fr/sante",                   # French government
            "https://www.govt.nz/travel",                  # New Zealand government
            "https://www.cdc.gov/dengue",                  # top-level domain is .gov
            "https://www.who.int/emergencies/x",           # .int
            "https://www.ecdc.europa.eu/en/dengue-monthly",  # parent domain on the allow-list
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
            "https://go.com/whatever",          # trap: go with no two-letter country code
            "https://gov.example.com/fake",     # trap: gov in front position does not count
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
        """Reorder only, never drop an item -- sources is also the verifier allow-list."""
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
        # Not one may be lost, or our own verifier could judge a really cited link invented
        assert len(merged) == 4
        # Within each class, the original search order is preserved
        assert [s.title for s in merged] == ["NEA", "MOH", "News", "Aggregator"]
