"""Scenario-based evaluation runner (eval harness): regression gate + failure library.

For every scenario in service/eval/scenarios.json, call /api/assess, /api/chat or
/api/destination **in-process** with a FastAPI TestClient (MOCK_MODE=true is forced, so
no real network request is made and **no search cost is incurred at all**), run the
declarative checks one at a time, and summarise passes/failures at the end.

Usage (Windows, from the project root):
    .venv\\Scripts\\python.exe service\\scripts\\eval_run.py
    .venv\\Scripts\\python.exe service\\scripts\\eval_run.py --json
    .venv\\Scripts\\python.exe service\\scripts\\eval_run.py --only healthy-young-adult,textbook-dengue
    .venv\\Scripts\\python.exe service\\scripts\\eval_run.py --scenarios path\\other-scenarios.json

A failing scenario has its request + response written out in full to
service/eval/failures/<id>.json (the failure library); leftovers from the previous run
are cleared before each run. Exit codes: 0 all passed, 1 some failed, 2 usage/file error.

Where this harness sits in the project's methodology: it is the one thing that "can tell
you whether the model needs retraining" -- the scenarios pin down **the behaviour that is
currently verified**, so any change to the model or the rules is seen here first.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIOS = SERVICE_ROOT / "eval" / "scenarios.json"
DEFAULT_FAILURES_DIR = SERVICE_ROOT / "eval" / "failures"

ENDPOINTS = {
    "assess": "/api/assess",
    "chat": "/api/chat",
    "destination": "/api/destination",
}

# Origin labels a source may carry (matches schemas.SourceOrigin; deliberately not imported)
SOURCE_ORIGINS = ("who", "search")

# Fields /api/destination must never carry: a location takes no part in scoring, no scores here
FORBIDDEN_SCORE_FIELDS = ("dengue", "worsening", "severe", "epi_week", "advice_source")

# Field names of the three models in the response (matches AssessmentResult)
MODEL_FIELDS = ("dengue", "worsening", "severe")

# The advice object's keys must serialise in this order (seek care first, see schemas.Advice)
ADVICE_ORDER = ["medical", "monitoring", "protection"]

LEVEL_ORDER = ("low", "medium", "high")

# z-comparison tolerance for scores_match_scenario: within the same process, the same
# coefficients and the same date, two assessments should give z values identical digit
# for digit; the tolerance is only there to absorb floating-point noise.
Z_TOLERANCE = 1e-6

# ---- Per-language "seek care / urgency" keyword tables (medical_urgency check) ----
# The keywords pin down wording that really exists in the current MOCK advice copy; real
# DeepSeek output should hit the same table -- output that misses it is exactly the kind
# worth putting in the failure library.
URGENCY_KEYWORDS: dict[str, list[str]] = {
    "zh-CN": ["就医", "就诊", "急诊"],
    "zh-TW": ["就醫", "就診", "急診"],
    "en": [
        "seek medical",
        "seek care",
        "medical care",
        "see a clinician",
        "medical review",
        "emergency department",
    ],
    "es": ["atención médica", "acuda", "consulte", "consulta médica"],
    "pt": [
        "procure atendimento",
        "atendimento médico",
        "procure um profissional",
        "avaliação médica",
        "pronto-socorro",
    ],
}

# ---- Overall-band labels a follow-up reply must mention (reply_mentions_tier check) ----
# Same as _MOCK_CHAT_TIER_LABELS in app/deepseek_client.py; deliberately **not** imported
# from the app package -- the harness pins down expected behaviour, so a change in the
# copy should turn red here rather than being silently followed.
TIER_LABELS: dict[str, dict[str, str]] = {
    "zh-CN": {"low": "较低", "medium": "中等", "high": "偏高"},
    "zh-TW": {"low": "較低", "medium": "中等", "high": "偏高"},
    "en": {"low": "low", "medium": "moderate", "high": "high"},
    "es": {"low": "bajo", "medium": "moderado", "high": "alto"},
    "pt": {"low": "baixo", "medium": "moderado", "high": "alto"},
}

# ---- Probability-wording detection (no_probability_language check) ----
# A string that contains both "a numeric percentage" and "probability-style wording"
# fails: the model is intercept-free / relative scoring, so any output of the
# "37% chance of infection" kind is copy that breaks the contract.
PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
PROBABILITY_RE = re.compile(
    r"概率|几率|機率|probabilit|probabilidad|probabilidade|chance|likelihood",
    re.IGNORECASE,
)

# ---- Citation link detection (sources_urls_allowed check) ----
# Again deliberately not imported from app.verifier: the harness checks the same
# invariant with an **independently written** regex and a stricter "exact equality"
# test -- every link in the reply must really be in this round's sources. Only when both
# sides pass do we know the invariant does not hold by coincidence in one implementation.
URL_RE = re.compile(r"https?://[^\s<>\"'）)】\[\]（(，。；、]+")
URL_TRAILING = ".,;:!?'\")]}>，。；！？、）】"

ADVICE_SOURCES = ("llm", "template")

# Chinese (including Traditional) language codes: keywords/labels match as substrings,
# Latin-script languages match on word boundaries
_CJK_LANGUAGES = ("zh-CN", "zh-TW")


# ---------- Infrastructure ----------


def build_client():
    """Build an in-process TestClient with MOCK_MODE forced and the feedback log off.

    Same routine as the client fixture in tests/test_pipeline.py: set the environment
    variables first, clear the settings cache, then import app. EVAL_LOG_PATH is emptied
    to keep harness traffic out of the real feedback data in data/assessments.jsonl.
    """
    os.environ["MOCK_MODE"] = "true"
    os.environ["EVAL_LOG_PATH"] = ""
    if str(SERVICE_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVICE_ROOT))

    # Squash app / httpx INFO logs: pipeline logs from dozens of scenarios drown the results
    for name in ("app", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)

    from app.config import get_settings

    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def load_scenarios(path: Path) -> list[dict]:
    """Read and sanity-check the scenario file: list of objects, unique ids, valid endpoint."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"场景文件顶层必须是 JSON 列表：{path}")
    seen: set[str] = set()
    for scenario in raw:
        if not isinstance(scenario, dict) or "id" not in scenario:
            raise ValueError("每个场景必须是含 id 的 JSON 对象")
        sid = scenario["id"]
        if sid in seen:
            raise ValueError(f"场景 id 重复：{sid}")
        seen.add(sid)
        endpoint = scenario.get("endpoint", "assess")
        if endpoint not in ENDPOINTS:
            raise ValueError(f"场景 {sid} 的 endpoint 非法：{endpoint}")
        if "request" not in scenario or "checks" not in scenario:
            raise ValueError(f"场景 {sid} 缺少 request 或 checks")
    return raw


def _request_language(scenario: dict) -> str:
    return scenario.get("request", {}).get("language", "zh-CN")


def _dig(body, path: str):
    """Look up a value by dotted path, e.g. "dengue.z" or "advice.medical.0".

    A list segment uses an integer index.
    """
    cur = body
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(path)
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def _context_tier(scenario: dict) -> str:
    """Follow-up scenarios: take the highest of the three model levels out of the context
    posted back by the front end (same as pipeline.overall_tier).
    """
    context = scenario.get("request", {}).get("context") or {}
    best = "low"
    for field in MODEL_FIELDS:
        block = context.get(field)
        if isinstance(block, dict):
            level = block.get("level")
            if level in LEVEL_ORDER and LEVEL_ORDER.index(level) > LEVEL_ORDER.index(best):
                best = level
    return best


def _gather_text_fields(body: dict) -> list[str]:
    """What no_probability_language scans: summary + each item of the three advice kinds."""
    texts: list[str] = []
    summary = body.get("summary")
    if isinstance(summary, str):
        texts.append(summary)
    advice = body.get("advice")
    if isinstance(advice, dict):
        for items in advice.values():
            if isinstance(items, list):
                texts.extend(s for s in items if isinstance(s, str))
    return texts


# ---------- Check implementations ----------
# Every check function: (check, scenario, status, body, runner) -> (ok, detail)
# detail must carry the **actual value** -- on a failure you can see straight away where
# the difference is; that is the point of the failure library.


def _check_status(check, scenario, status, body, runner):
    expect = check.get("expect")
    return status == expect, f"expected HTTP {expect}, got {status}"


def _check_level(check, scenario, status, body, runner):
    model = check.get("model")
    expect_in = check.get("expect_in", [])
    if not isinstance(body, dict) or not isinstance(body.get(model), dict):
        return False, f"response has no model block '{model}' (HTTP {status})"
    block = body[model]
    got = block.get("level")
    ok = got in expect_in
    return ok, f"{model}.level expected in {expect_in}, got '{got}' (score={block.get('score')})"


def _check_score_between(check, scenario, status, body, runner):
    model = check.get("model")
    lo, hi = check.get("lo"), check.get("hi")
    if not isinstance(body, dict) or not isinstance(body.get(model), dict):
        return False, f"response has no model block '{model}' (HTTP {status})"
    got = body[model].get("score")
    ok = isinstance(got, (int, float)) and lo <= got <= hi
    return ok, f"{model}.score expected in [{lo}, {hi}], got {got}"


def _check_warning_signs(check, scenario, status, body, runner):
    expect = check.get("expect", [])
    if not isinstance(body, dict) or not isinstance(body.get("warning_signs"), list):
        return False, f"response has no warning_signs list (HTTP {status})"
    got = body["warning_signs"]
    ok = set(got) == set(expect)
    return ok, f"warning_signs expected {sorted(expect)}, got {sorted(got)}"


def _check_exposure_level(check, scenario, status, body, runner):
    expect = check.get("expect")
    try:
        got = _dig(body, "exposure_context.level")
    except (KeyError, TypeError, ValueError, IndexError):
        return False, f"response has no exposure_context.level (HTTP {status})"
    return got == expect, f"exposure_context.level expected '{expect}', got '{got}'"


def _check_field_equals(check, scenario, status, body, runner):
    path, expect = check.get("path"), check.get("expect")
    try:
        got = _dig(body, path)
    except (KeyError, TypeError, ValueError, IndexError):
        return False, f"path '{path}' not found in response (HTTP {status})"
    return got == expect, f"{path} expected {expect!r}, got {got!r}"


def _check_advice_order(check, scenario, status, body, runner):
    if not isinstance(body, dict) or not isinstance(body.get("advice"), dict):
        return False, f"response has no advice object (HTTP {status})"
    got = list(body["advice"].keys())
    return got == ADVICE_ORDER, f"advice keys expected {ADVICE_ORDER}, got {got}"


def _check_medical_urgency(check, scenario, status, body, runner):
    language = _request_language(scenario)
    keywords = URGENCY_KEYWORDS.get(language)
    if keywords is None:
        return False, f"no urgency lexicon for language '{language}'"
    try:
        items = _dig(body, "advice.medical")
    except (KeyError, TypeError, ValueError, IndexError):
        return False, f"response has no advice.medical (HTTP {status})"
    if not isinstance(items, list) or not items:
        return False, f"advice.medical is empty or not a list: {items!r}"
    for item in items:
        text = str(item).lower()
        if any(k.lower() in text for k in keywords):
            return True, f"matched urgency keyword in advice.medical ({language})"
    return False, (
        f"no advice.medical item matched urgency lexicon {keywords} "
        f"(language={language}); items={items!r}"
    )


def _check_no_probability_language(check, scenario, status, body, runner):
    if not isinstance(body, dict):
        return False, f"response body is not an object (HTTP {status})"
    offending = [
        text
        for text in _gather_text_fields(body)
        if PERCENT_RE.search(text) and PROBABILITY_RE.search(text)
    ]
    if offending:
        return False, f"probability-claim wording found in advice/summary: {offending!r}"
    return True, "no probability-claim wording in advice/summary"


def _check_explanations_present(check, scenario, status, body, runner):
    if not isinstance(body, dict) or not isinstance(body.get("explanations"), dict):
        return False, f"response has no explanations object (HTTP {status})"
    explanations = body["explanations"]
    missing = set(MODEL_FIELDS) - set(explanations)
    if missing:
        return False, f"explanations missing models: {sorted(missing)}"
    for field in MODEL_FIELDS:
        items = explanations[field]
        if not isinstance(items, list) or not (1 <= len(items) <= 5):
            return False, f"explanations.{field} expected 1-5 entries, got {len(items) if isinstance(items, list) else items!r}"
        try:
            magnitudes = [abs(float(i["contribution"])) for i in items]
        except (KeyError, TypeError, ValueError):
            return False, f"explanations.{field} entries lack numeric 'contribution': {items!r}"
        if magnitudes != sorted(magnitudes, reverse=True):
            return False, (
                f"explanations.{field} not sorted by |contribution| desc: {magnitudes}"
            )
    return True, "explanations: 1-5 entries per model, sorted by |contribution| desc"


def _check_reply_nonempty(check, scenario, status, body, runner):
    reply = body.get("reply") if isinstance(body, dict) else None
    ok = isinstance(reply, str) and bool(reply.strip())
    return ok, f"reply expected non-empty string, got {reply!r}"


def _check_reply_mentions_tier(check, scenario, status, body, runner):
    language = _request_language(scenario)
    labels = TIER_LABELS.get(language)
    if labels is None:
        return False, f"no tier labels for language '{language}'"
    tier = _context_tier(scenario)
    label = labels[tier]
    reply = body.get("reply") if isinstance(body, dict) else None
    if not isinstance(reply, str):
        return False, f"reply is not a string: {reply!r}"
    if language in _CJK_LANGUAGES:
        ok = label in reply
    else:
        ok = re.search(rf"\b{re.escape(label)}\b", reply, re.IGNORECASE) is not None
    return ok, (
        f"reply expected to mention overall tier '{tier}' as '{label}' "
        f"(language={language}); reply={reply!r}"
    )


def _citable_text(body: dict) -> str:
    """All fields scanned for links: the reply, destination findings, destination advice.

    /api/chat only has reply; for /api/destination "the part the model wrote" is
    recent_findings, and although the advice is fixed copy it gets scanned too -- the
    invariant being pinned down is "every link appearing in the response must come from
    sources", not "every link in reply must come from sources".
    """
    parts = [str(body.get("reply") or "")]
    findings = body.get("recent_findings")
    if isinstance(findings, list):
        parts += [str(f) for f in findings]
    advice = body.get("advice")
    if isinstance(advice, dict):
        for items in advice.values():
            if isinstance(items, list):
                parts += [str(i) for i in items]
    return "\n".join(parts)


def _check_sources_urls_allowed(check, scenario, status, body, runner):
    """Every link in the response must appear in this round's sources (optionally with a
    bound on the number of sources).

    This is the end-to-end gate for "no fabricated citations": sources is what the API
    really returned, and the generated text may only cite those. min_sources /
    max_sources pin down the two scenarios "asking about a location should produce
    sources" and "not asking about a location should produce none".
    """
    if not isinstance(body, dict):
        return False, f"response body is not an object (HTTP {status})"
    sources = body.get("sources")
    if not isinstance(sources, list):
        return False, f"response has no sources list (HTTP {status}); body keys={sorted(body)}"

    allowed = [str(s.get("url", "")) for s in sources if isinstance(s, dict)]
    minimum, maximum = check.get("min_sources"), check.get("max_sources")
    if minimum is not None and len(sources) < minimum:
        return False, f"expected at least {minimum} source(s), got {len(sources)}"
    if maximum is not None and len(sources) > maximum:
        return False, f"expected at most {maximum} source(s), got {len(sources)}: {allowed}"

    cited = [u.rstrip(URL_TRAILING) for u in URL_RE.findall(_citable_text(body))]
    unlisted = [u for u in cited if u not in allowed]
    if unlisted:
        return False, f"response cites URL(s) absent from sources: {unlisted}; sources={allowed}"
    return True, (
        f"{len(cited)} cited URL(s) all present in {len(sources)} source(s): {allowed}"
    )


def _check_sources_origins(check, scenario, status, body, runner):
    """Every source must state where it came from, and the expected origins must appear.

    Provenance is the whole point of this feature: WHO notices and web search results
    differ in recency and trustworthiness, and the front end has to label them
    separately. With a source that has no origin, the user cannot judge how solid it is.
    """
    if not isinstance(body, dict) or not isinstance(body.get("sources"), list):
        return False, f"response has no sources list (HTTP {status})"
    sources = body["sources"]
    got = [s.get("origin") for s in sources if isinstance(s, dict)]
    bad = [o for o in got if o not in SOURCE_ORIGINS]
    if bad:
        return False, f"sources carry unknown origin label(s): {bad}; expected one of {list(SOURCE_ORIGINS)}"
    expected = check.get("expect_origins") or []
    missing = [o for o in expected if o not in got]
    if missing:
        return False, f"expected origin(s) {missing} among sources, got {got}"
    return True, f"{len(got)} source(s) labelled: {got}"


def _check_search_count(check, scenario, status, body, runner):
    """How many searches this round really made.

    0 is what pins down the rule "no location means not a cent is spent".
    """
    expect = check.get("expect")
    got = body.get("search_count") if isinstance(body, dict) else None
    maximum = check.get("max")
    if expect is not None:
        return got == expect, f"search_count expected {expect}, got {got!r}"
    if maximum is not None:
        ok = isinstance(got, int) and got <= maximum
        return ok, f"search_count expected <= {maximum}, got {got!r}"
    return False, "search_count check needs 'expect' or 'max'"


def _check_no_model_scores(check, scenario, status, body, runner):
    """A destination lookup must never carry scores: a location never takes part in
    scoring, so inventing a "destination risk score" would be a lie.
    """
    if not isinstance(body, dict):
        return False, f"response body is not an object (HTTP {status})"
    present = [f for f in FORBIDDEN_SCORE_FIELDS if f in body]
    if present:
        return False, f"destination response must carry no scores, but has: {present}"
    return True, f"no scoring fields present (checked {list(FORBIDDEN_SCORE_FIELDS)})"


def _check_advice_source(check, scenario, status, body, runner):
    """advice_source must say truthfully whether this text was written by the model or
    came from the template fallback.
    """
    expect = check.get("expect")
    if expect not in ADVICE_SOURCES:
        return False, f"advice_source check needs expect in {list(ADVICE_SOURCES)}, got {expect!r}"
    got = body.get("advice_source") if isinstance(body, dict) else None
    return got == expect, f"advice_source expected {expect!r}, got {got!r}"


def _check_scores_match_scenario(check, scenario, status, body, runner):
    ref_id = check.get("ref")
    ref = runner.by_id.get(ref_id)
    if ref is None:
        return False, f"referenced scenario '{ref_id}' not found in scenarios file"
    ref_status, ref_body = runner.call(ref)
    if ref_status != 200 or not isinstance(ref_body, dict):
        return False, f"referenced scenario '{ref_id}' did not return HTTP 200 ({ref_status})"
    if status != 200 or not isinstance(body, dict):
        return False, f"this scenario did not return HTTP 200 ({status})"

    diffs = []
    for field in MODEL_FIELDS:
        z_here = body.get(field, {}).get("z")
        z_ref = ref_body.get(field, {}).get("z")
        if (
            not isinstance(z_here, (int, float))
            or not isinstance(z_ref, (int, float))
            or abs(z_here - z_ref) > Z_TOLERANCE
        ):
            diffs.append(f"{field}.z: {z_here} != {z_ref}")
    if diffs:
        return False, f"z values differ from scenario '{ref_id}': " + "; ".join(diffs)
    return True, f"all three z values match scenario '{ref_id}'"


CHECKS = {
    "status": _check_status,
    "level": _check_level,
    "score_between": _check_score_between,
    "warning_signs": _check_warning_signs,
    "exposure_level": _check_exposure_level,
    "field_equals": _check_field_equals,
    "advice_order": _check_advice_order,
    "medical_urgency": _check_medical_urgency,
    "no_probability_language": _check_no_probability_language,
    "explanations_present": _check_explanations_present,
    "reply_nonempty": _check_reply_nonempty,
    "reply_mentions_tier": _check_reply_mentions_tier,
    "sources_urls_allowed": _check_sources_urls_allowed,
    "sources_origins": _check_sources_origins,
    "search_count": _check_search_count,
    "no_model_scores": _check_no_model_scores,
    "advice_source": _check_advice_source,
    "scores_match_scenario": _check_scores_match_scenario,
}


# ---------- Runner ----------


class ScenarioRunner:
    """Run scenario requests on demand and cache the responses.

    scores_match_scenario needs to reference other scenarios.
    """

    def __init__(self, client, scenarios: list[dict]) -> None:
        self.client = client
        self.by_id = {s["id"]: s for s in scenarios}
        self._cache: dict[str, tuple[int, object]] = {}

    def call(self, scenario: dict) -> tuple[int, object]:
        sid = scenario["id"]
        if sid not in self._cache:
            path = ENDPOINTS[scenario.get("endpoint", "assess")]
            resp = self.client.post(path, json=scenario["request"])
            try:
                parsed = resp.json()
            except ValueError:
                parsed = None
            self._cache[sid] = (resp.status_code, parsed)
        return self._cache[sid]

    def evaluate(self, scenario: dict) -> dict:
        status, body = self.call(scenario)
        checks = []
        for check in scenario.get("checks", []):
            ctype = check.get("type")
            fn = CHECKS.get(ctype)
            if fn is None:
                ok, detail = False, f"unknown check type '{ctype}'"
            else:
                try:
                    ok, detail = fn(check, scenario, status, body, self)
                except Exception as exc:  # a crashing check fails, it never kills the run
                    ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
            checks.append({"type": ctype, "ok": bool(ok), "detail": detail})
        return {
            "id": scenario["id"],
            "endpoint": scenario.get("endpoint", "assess"),
            "description": scenario.get("description", ""),
            "status": status,
            "passed": all(c["ok"] for c in checks),
            "checks": checks,
        }


def _clear_failures_dir(failures_dir: Path) -> None:
    """Create the directory and clear last round's dumps -- it keeps only this run's cases."""
    failures_dir.mkdir(parents=True, exist_ok=True)
    for stale in failures_dir.glob("*.json"):
        stale.unlink()


def _dump_failure(failures_dir: Path, scenario: dict, result: dict, body) -> Path:
    """Dump a failure case: request + response + failed checks, enough to review offline."""
    record = {
        "id": scenario["id"],
        "description": scenario.get("description", ""),
        "endpoint": scenario.get("endpoint", "assess"),
        "request": scenario["request"],
        "status": result["status"],
        "response": body,
        "failed_checks": [c for c in result["checks"] if not c["ok"]],
    }
    path = failures_dir / f"{scenario['id']}.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def run_scenarios(
    scenarios: list[dict],
    only: list[str] | None = None,
    failures_dir: Path = DEFAULT_FAILURES_DIR,
) -> dict:
    """Run the (optionally filtered) set of scenarios and return a summary dict."""
    selected = scenarios
    if only:
        by_id = {s["id"]: s for s in scenarios}
        unknown = [sid for sid in only if sid not in by_id]
        if unknown:
            raise ValueError(f"--only 指定了不存在的场景 id：{unknown}")
        selected = [by_id[sid] for sid in only]

    _clear_failures_dir(failures_dir)

    results = []
    dumps: list[str] = []
    with build_client() as client:
        runner = ScenarioRunner(client, scenarios)
        for scenario in selected:
            result = runner.evaluate(scenario)
            results.append(result)
            if not result["passed"]:
                _, body = runner.call(scenario)
                dumps.append(str(_dump_failure(failures_dir, scenario, result, body)))

    passed = sum(1 for r in results if r["passed"])
    return {
        "total": len(results),
        "passed": passed,
        "failed": [r["id"] for r in results if not r["passed"]],
        "failure_dumps": dumps,
        "results": results,
    }


# ---------- Output ----------


def _symbols() -> tuple[str, str]:
    """Windows console encoding fallback: drop to [PASS]/[FAIL] if emoji cannot encode."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✅❌".encode(encoding)
        return "✅", "❌"
    except (UnicodeEncodeError, LookupError):
        return "[PASS]", "[FAIL]"


def _safe_print(text: str) -> None:
    """Print lines containing Chinese/emoji without crashing on the console encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding))


def print_report(summary: dict, failures_dir: Path) -> None:
    pass_mark, fail_mark = _symbols()
    for result in summary["results"]:
        mark = pass_mark if result["passed"] else fail_mark
        _safe_print(f"{mark} {result['id']}")
        if not result["passed"]:
            for check in result["checks"]:
                if not check["ok"]:
                    _safe_print(f"    - {check['type']}: {check['detail']}")
    _safe_print(f"passed {summary['passed']}/{summary['total']}")
    if summary["failed"]:
        _safe_print(f"failure library: {len(summary['failed'])} dump(s) in {failures_dir}")
        for path in summary["failure_dumps"]:
            _safe_print(f"    {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="登革热风险服务场景化评测运行器")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON 结果")
    parser.add_argument(
        "--only",
        default="",
        help="只运行指定场景，逗号分隔的 id 列表，如 --only a,b",
    )
    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIOS),
        help="场景文件路径（默认 service/eval/scenarios.json）",
    )
    parser.add_argument(
        "--failures-dir",
        default=str(DEFAULT_FAILURES_DIR),
        help="失败案例转储目录（默认 service/eval/failures）",
    )
    args = parser.parse_args(argv)

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.is_file():
        print(f"场景文件不存在：{scenarios_path}", file=sys.stderr)
        return 2
    try:
        scenarios = load_scenarios(scenarios_path)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"场景文件非法：{exc}", file=sys.stderr)
        return 2

    only = [sid.strip() for sid in args.only.split(",") if sid.strip()]
    try:
        summary = run_scenarios(
            scenarios, only=only or None, failures_dir=Path(args.failures_dir)
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        _safe_print(json.dumps({"scenarios": str(scenarios_path), **summary}, ensure_ascii=True, indent=2))
    else:
        print_report(summary, Path(args.failures_dir))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
