"""Integration tests for the two "agentic" paths: advice fallback and the chat tool loop.

What these two paths share is that **the behaviour on failure** is the real subject, so
most of the tests here manufacture failures: the upstream blows up, the model emits a
violation, the model invents a link. One or two happy-path tests are enough.

Everything runs under MOCK_MODE; where the real branch is needed, monkeypatch replaces the
client methods, and still no network request is made.
"""

import pytest

from app.verifier import verify_chat_reply

WHO_PREFIX = "https://www.who.int/emergencies/disease-outbreak-news/item/"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "true")
    from app.config import get_settings

    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


@pytest.fixture()
def live_client(monkeypatch):
    """MOCK_MODE=false: the real branch, but every outbound call is monkeypatched away."""
    monkeypatch.setenv("MOCK_MODE", "false")
    from app.config import get_settings

    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


ALL_NO_SYMPTOMS = {
    c: "no"
    for c in (
        "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA",
        "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N",
        "LEUCOPENIA", "LACO", "DOR_RETRO",
    )
}
ALL_NO_COMORB = {
    c: "no"
    for c in ("DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL", "HIPERTENSA",
              "ACIDO_PEPT", "AUTO_IMUNE")
}


def form(**overrides) -> dict:
    body = {
        "age": 35,
        "sex": "F",
        "day_ill": 3,
        "symptoms": {**ALL_NO_SYMPTOMS, "FEBRE": "yes", "CEFALEIA": "yes"},
        "comorbidities": ALL_NO_COMORB,
        "language": "en",
    }
    body.update(overrides)
    return body


def tier_of(body: dict) -> str:
    """Highest of the three model levels in the response (same as pipeline.overall_tier)."""
    return max(
        (body[f]["level"] for f in ("dengue", "worsening", "severe")),
        key=["low", "medium", "high"].index,
    )


def chat(**overrides) -> dict:
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


# ================= Advice: verify -> re-ask -> fallback =================


def test_mock_assessment_reports_template_source(client):
    body = client.post("/api/assess", json=form()).json()
    assert body["advice_source"] == "template"


@pytest.mark.parametrize("language", ["zh-CN", "zh-TW", "en", "es", "pt"])
def test_advice_failure_returns_200_with_template_not_502(live_client, monkeypatch, language):
    """**A deliberate behaviour change**: a failure in the advice step no longer fails the
    whole assessment.

    The scores are computed locally, and they are the genuinely valuable part of this
    service. Trading a result the user already has for a 502, just because one paragraph of
    natural language could not be fetched, is a very bad exchange.
    """
    from app.deepseek_client import DeepSeekClient, DeepSeekError, fallback_advice

    async def boom(*args, **kwargs):
        raise DeepSeekError("upstream blew up")

    monkeypatch.setattr(DeepSeekClient, "chat_json", boom)
    resp = live_client.post("/api/assess", json=form(language=language))

    assert resp.status_code == 200
    body = resp.json()
    assert body["advice_source"] == "template"
    # The scores still come back, and they really were computed
    assert body["dengue"]["score"] > 0
    assert body["explanations"]["dengue"]
    # The copy is that one shared template; there is no second set of text
    expected = fallback_advice(language, tier_of(body))
    assert body["advice"] == expected["advice"]
    assert body["summary"] == expected["summary"]


def test_clean_llm_advice_is_reported_as_llm(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient, fallback_advice

    good = fallback_advice("en", "medium")

    async def fake_chat_json(self, system, user, purpose="advice", **kwargs):
        if purpose == "features":
            return {"infer": {}}
        return good

    monkeypatch.setattr(DeepSeekClient, "chat_json", fake_chat_json)
    body = live_client.post("/api/assess", json=form()).json()
    assert body["advice_source"] == "llm"


def test_violating_advice_is_retried_once_then_accepted(live_client, monkeypatch):
    """First reply carries a dosage, second is clean: the second is used and marked llm."""
    from app.deepseek_client import DeepSeekClient, fallback_advice

    calls: list[str] = []
    dirty = fallback_advice("en", "medium")
    dirty = {
        "summary": dirty["summary"],
        "advice": {
            "medical": ["Take 500 mg of paracetamol every 6 hours.", *dirty["advice"]["medical"]],
            "monitoring": list(dirty["advice"]["monitoring"]),
            "protection": list(dirty["advice"]["protection"]),
        },
    }
    clean = fallback_advice("en", "medium")

    async def fake_chat_json(self, system, user, purpose="advice", **kwargs):
        if purpose == "features":
            return {"infer": {}}
        calls.append(user)
        return dirty if len(calls) == 1 else clean

    monkeypatch.setattr(DeepSeekClient, "chat_json", fake_chat_json)
    body = live_client.post("/api/assess", json=form()).json()

    assert len(calls) == 2
    # The second prompt must carry the violation notice, or the model cannot know what to fix
    assert "[dosage]" in calls[1]
    assert "violated" in calls[1]
    assert body["advice_source"] == "llm"
    assert "500 mg" not in " ".join(body["advice"]["medical"])


def test_advice_violating_twice_falls_back_to_template(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient, fallback_advice

    calls = []
    bad = {
        "summary": "You have a 42% probability of infection.",
        "advice": {
            "medical": ["Take 500 mg of paracetamol every 6 hours."],
            "monitoring": ["Watch your temperature and drink water with the family."],
            "protection": ["Use the repellent and the bed net."],
        },
    }

    async def fake_chat_json(self, system, user, purpose="advice", **kwargs):
        if purpose == "features":
            return {"infer": {}}
        calls.append(user)
        return bad

    monkeypatch.setattr(DeepSeekClient, "chat_json", fake_chat_json)
    body = live_client.post("/api/assess", json=form()).json()

    assert len(calls) == 2  # first call + one re-ask, not an unbounded retry loop
    assert body["advice_source"] == "template"
    assert "500 mg" not in " ".join(body["advice"]["medical"])
    assert "42%" not in body["summary"]
    assert body["advice"] == fallback_advice("en", tier_of(body))["advice"]


def test_structurally_broken_advice_falls_back(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient

    async def fake_chat_json(self, system, user, purpose="advice", **kwargs):
        if purpose == "features":
            return {"infer": {}}
        return {"summary": "ok", "advice": {"medical": "not a list"}}

    monkeypatch.setattr(DeepSeekClient, "chat_json", fake_chat_json)
    body = live_client.post("/api/assess", json=form()).json()
    assert body["advice_source"] == "template"


# ================= Follow-up: tool loop, sources, fabricated links =================


def test_mock_chat_without_a_location_calls_no_tool(client):
    body = client.post("/api/chat", json=chat()).json()
    assert body["sources"] == []
    assert "http" not in body["reply"]


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("I am flying to Singapore next month, is dengue a risk?", "en"),
        ("下个月我要去新加坡，需要注意登革热吗？", "zh-CN"),
        ("Voy a viajar a Brasil, ¿hay riesgo de dengue?", "es"),
        ("Vou para a Tailândia; devo me preocupar com dengue?", "pt"),
    ],
)
def test_mock_chat_with_a_location_returns_citable_sources(client, question, language):
    body = client.post("/api/chat", json=chat(question=question, language=language)).json()

    assert body["reply"].strip()
    assert body["sources"], "a place name mentioned must come with a source"
    origins = {s["origin"] for s in body["sources"]}
    assert origins == {"who", "search"}, "both layers must appear, each labelled with its origin"
    for source in body["sources"]:
        assert set(source) == {"title", "date", "url", "origin", "authority"}
        assert source["url"].startswith("http")
        assert source["authority"] in {"official", "other"}
        if source["origin"] == "who":
            assert source["url"].startswith(WHO_PREFIX)
            assert source["authority"] == "official", "a WHO notice is official by definition"
    # Every link that appears in the reply must be in sources -- that is the invariant
    assert verify_chat_reply(body["reply"], language, [s["url"] for s in body["sources"]]) == []


def test_mock_chat_reply_cites_a_url_that_really_came_from_the_tool(client):
    body = client.post(
        "/api/chat", json=chat(question="I am moving to Singapore.")
    ).json()
    urls = {s["url"] for s in body["sources"]}
    cited = [u for u in urls if u in body["reply"]]
    assert cited, f"the demo reply should cite a link the tool returned: {body['reply']!r}"


def test_language_names_in_the_prompt_are_not_mistaken_for_places(client):
    """The "Portuguese" / "Spanish" inside the prompt must not be read as Portugal / Spain.

    The MOCK decision looks only at the user's own text, not at the whole prompt -- otherwise
    every Portuguese-speaking user would inexplicably receive a travel note about Portugal.
    """
    for language in ("pt", "es"):
        body = client.post(
            "/api/chat", json=chat(language=language, question="¿Qué significa mi puntuación?")
        ).json()
        assert body["sources"] == []


def test_history_mentioning_a_place_still_triggers_the_tool(client):
    body = client.post(
        "/api/chat",
        json=chat(
            question="And what about the rainy season there?",
            history=[{"role": "user", "content": "I am going to Thailand in July."}],
        ),
    ).json()
    assert body["sources"]


def test_fabricated_url_forces_the_localised_fallback(live_client, monkeypatch):
    """The model invents a who.int link: if it does so in both rounds, the reply must be
    swapped for the fallback sentence and sources must be emptied.

    The question deliberately names no place, so this takes the **function tool** path (no
    location means no search). The same invariant on the search path is guarded separately
    by tests/test_search.py.
    """
    from app.deepseek_client import DeepSeekClient
    from app.pipeline import _UNRELIABLE_REPLY

    calls = []

    async def fake(self, system, messages, tools, tool_executor, **kwargs):
        calls.append(messages)
        return {
            "reply": "You are fine, see https://www.who.int/made-up-page for details.",
            "tool_results": [],
        }

    monkeypatch.setattr(DeepSeekClient, "chat_with_tools", fake)
    resp = live_client.post("/api/chat", json=chat(question="Am I at risk?"))

    assert resp.status_code == 200
    body = resp.json()
    assert len(calls) == 2  # first call + one re-ask carrying the violation notice
    assert body["reply"] == _UNRELIABLE_REPLY["en"]
    assert body["sources"] == []
    assert body["search_count"] == 0  # no location = no money spent on search
    assert "who.int" not in body["reply"]


def test_retry_prompt_carries_the_violation_message(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient

    seen = []

    async def fake(self, system, messages, tools, tool_executor, **kwargs):
        seen.append(messages)
        return {"reply": "See https://example.org/fake", "tool_results": []}

    monkeypatch.setattr(DeepSeekClient, "chat_with_tools", fake)
    live_client.post("/api/chat", json=chat())

    retry = seen[1]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert "[fabricated_url]" in retry[-1]["content"]
    assert "example.org/fake" in retry[-1]["content"]


def test_a_url_that_the_tool_did_return_is_accepted(live_client, monkeypatch):
    """The other direction: a link the tool really did return passes through into sources."""
    from app.deepseek_client import DeepSeekClient

    url = WHO_PREFIX + "2024-DON518"
    tool_result = {
        "location": "Singapore",
        "matched": True,
        "endemicity": "high",
        "season_note": "Year-round.",
        "who_notices": [{"title": "Dengue - Global situation", "date": "2024-05-30", "url": url}],
        "lookup_failed": False,
    }

    async def fake(self, system, messages, tools, tool_executor, **kwargs):
        return {
            "reply": f"Singapore is highly endemic. See {url}",
            "tool_results": [
                {"name": "lookup_dengue_context", "arguments": {"location": "Singapore"}, "result": tool_result}
            ],
        }

    monkeypatch.setattr(DeepSeekClient, "chat_with_tools", fake)
    body = live_client.post("/api/chat", json=chat()).json()

    assert body["sources"] == [
        {
            "title": "Dengue - Global situation",
            "date": "2024-05-30",
            "url": url,
            "origin": "who",
            "authority": "official",
        }
    ]
    assert url in body["reply"]


def test_sources_are_deduplicated_across_tool_calls(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient

    url = WHO_PREFIX + "2024-DON518"
    notice = {"title": "Dengue - Global situation", "date": "2024-05-30", "url": url}
    result = {"who_notices": [notice, notice], "lookup_failed": False}

    async def fake(self, system, messages, tools, tool_executor, **kwargs):
        return {
            "reply": "Both places are endemic.",
            "tool_results": [
                {"name": "lookup_dengue_context", "arguments": {"location": "Brazil"}, "result": result},
                {"name": "lookup_dengue_context", "arguments": {"location": "Thailand"}, "result": result},
            ],
        }

    monkeypatch.setattr(DeepSeekClient, "chat_with_tools", fake)
    body = live_client.post("/api/chat", json=chat()).json()
    assert len(body["sources"]) == 1


def test_tool_executor_rejects_unknown_tool_names():
    from app.pipeline import _make_tool_executor

    collected: list[dict] = []
    execute = _make_tool_executor(collected)
    result = execute("rm_minus_rf", {"location": "Singapore"})
    assert result["lookup_failed"] is True
    assert "unknown tool" in result["error"]
    assert collected == []


def test_tool_executor_returns_the_lookup_payload(client):
    from app.pipeline import _make_tool_executor

    collected: list[dict] = []
    execute = _make_tool_executor(collected)
    result = execute("lookup_dengue_context", {"location": "新加坡"})
    assert result["location"] == "Singapore"
    assert collected == [result]


# ================= Client: the tools loop itself =================


@pytest.mark.anyio
async def test_tool_loop_executes_calls_and_feeds_results_back(live_client, monkeypatch):
    """Two rounds: first a tool call is requested, then its result feeds the final reply."""
    from app.deepseek_client import DeepSeekClient

    sent: list[list[dict]] = []
    replies = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_dengue_context",
                        "arguments": '{"location": "Singapore"}',
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Singapore is highly endemic."},
    ]

    async def fake_request_message(self, client, messages, purpose, json_mode, temperature, tools=None):
        sent.append([dict(m) for m in messages])
        return replies[len(sent) - 1]

    monkeypatch.setattr(DeepSeekClient, "_request_message", fake_request_message)

    executed = []

    def executor(name, args):
        executed.append((name, args))
        return {"location": "Singapore", "endemicity": "high", "who_notices": []}

    outcome = await DeepSeekClient().chat_with_tools(
        "system", [{"role": "user", "content": "Going to Singapore"}], [], executor
    )

    assert executed == [("lookup_dengue_context", {"location": "Singapore"})]
    assert outcome["reply"] == "Singapore is highly endemic."
    assert outcome["tool_results"][0]["name"] == "lookup_dengue_context"
    # The second round's context must carry the assistant tool_calls and the tool result
    roles = [m["role"] for m in sent[1]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "Singapore" in sent[1][-1]["content"]


@pytest.mark.anyio
async def test_tool_loop_stops_after_max_rounds(live_client, monkeypatch):
    """Model keeps asking for tools: exhausting the rounds must force an answer, not a loop."""
    from app.deepseek_client import DeepSeekClient

    tool_call_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_n",
                "type": "function",
                "function": {"name": "lookup_dengue_context", "arguments": '{"location": "Brazil"}'},
            }
        ],
    }
    rounds = []
    final = []

    async def fake_request_message(self, client, messages, purpose, json_mode, temperature, tools=None):
        rounds.append(tools)
        return tool_call_message

    async def fake_request(self, client, messages, purpose, json_mode, temperature):
        final.append([dict(m) for m in messages])
        return "Based on what I found, Brazil is highly endemic."

    monkeypatch.setattr(DeepSeekClient, "_request_message", fake_request_message)
    monkeypatch.setattr(DeepSeekClient, "_request", fake_request)

    outcome = await DeepSeekClient().chat_with_tools(
        "system", [{"role": "user", "content": "Brazil?"}], [{"type": "function"}],
        lambda name, args: {"ok": True}, max_rounds=2,
    )

    assert len(rounds) == 2
    assert len(outcome["tool_results"]) == 2
    assert outcome["reply"].startswith("Based on what I found")
    # The closing call offers no tools and explicitly asks for "answer with what you have"
    assert "Answer the user now" in final[0][-1]["content"]


@pytest.mark.anyio
async def test_tool_failure_becomes_data_not_a_crash(live_client, monkeypatch):
    from app.deepseek_client import DeepSeekClient

    replies = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup_dengue_context", "arguments": "not json"},
                }
            ],
        },
        {"role": "assistant", "content": "I could not look that up."},
    ]
    seen = []

    async def fake_request_message(self, client, messages, purpose, json_mode, temperature, tools=None):
        seen.append(messages)
        return replies[len(seen) - 1]

    monkeypatch.setattr(DeepSeekClient, "_request_message", fake_request_message)

    def angry_executor(name, args):
        raise RuntimeError("boom")

    outcome = await DeepSeekClient().chat_with_tools(
        "system", [{"role": "user", "content": "?"}], [], angry_executor
    )
    result = outcome["tool_results"][0]
    assert result["arguments"] == {}  # invalid JSON arguments are treated as empty arguments
    assert result["result"]["lookup_failed"] is True
    assert outcome["reply"] == "I could not look that up."


@pytest.fixture()
def anyio_backend():
    return "asyncio"
