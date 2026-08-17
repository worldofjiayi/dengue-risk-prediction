"""Tests for the scenario-based evaluation runner (scripts/eval_run.py).

Meta-testing idea: the harness itself is under test too --
  1. the shipped scenarios.json must be all green (that is the regression gate baseline);
  2. a deliberately failing scenario must produce exit code 1, a failure dump file, and
     output carrying the actual value;
  3. scores_match_scenario must be able to see through a mismatch in z;
  4. the --only filter and the --json output must work.

Everything runs in-process under MOCK_MODE, with no real network requests.
"""

import json
from pathlib import Path

import pytest

from scripts.eval_run import DEFAULT_SCENARIOS, load_scenarios, main

SERVICE_ROOT = Path(__file__).resolve().parent.parent

SYMPTOM_CODES = (
    "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA",
    "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N",
    "LEUCOPENIA", "LACO", "DOR_RETRO",
)
COMORB_CODES = (
    "DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL",
    "HIPERTENSA", "ACIDO_PEPT", "AUTO_IMUNE",
)
EXPOSURE_CODES = ("FEVER_CLUSTER", "CONFIRMED_CASE", "OUTBREAK_TRAVEL")


@pytest.fixture(autouse=True)
def _mock_mode(monkeypatch):
    """Runner writes os.environ directly; monkeypatch sets it first so teardown restores it."""
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("EVAL_LOG_PATH", "")


def full_form(yes_symptoms=(), **overrides) -> dict:
    """Build a questionnaire request body with every key present."""
    body = {
        "age": 25,
        "sex": "M",
        "day_ill": 0,
        "symptoms": {c: ("yes" if c in yes_symptoms else "no") for c in SYMPTOM_CODES},
        "comorbidities": {c: "no" for c in COMORB_CODES},
        "exposure": {c: "no" for c in EXPOSURE_CODES},
        "language": "zh-CN",
    }
    body.update(overrides)
    return body


def write_scenarios(tmp_path: Path, scenarios: list[dict]) -> Path:
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(scenarios, ensure_ascii=False), encoding="utf-8")
    return path


# ---------- Shipped scenarios: the regression gate baseline ----------


def test_shipped_scenarios_all_pass(tmp_path, capsys):
    """Meta-test: the shipped scenarios.json must all pass (exit code 0, no failure dumps)."""
    failures = tmp_path / "failures"
    assert main(["--failures-dir", str(failures)]) == 0

    out = capsys.readouterr().out
    assert "passed" in out
    assert "failure library" not in out
    # When all green the failures directory exists but is empty
    assert failures.is_dir()
    assert list(failures.glob("*.json")) == []


def test_shipped_scenarios_file_is_valid_and_broad():
    """The scenario file is itself valid and broad enough: three endpoints, 20-35 scenarios."""
    scenarios = load_scenarios(DEFAULT_SCENARIOS)
    assert 20 <= len(scenarios) <= 35
    endpoints = {s.get("endpoint", "assess") for s in scenarios}
    assert endpoints == {"assess", "chat", "destination"}
    # All five languages must be covered by assess scenarios
    languages = {s["request"].get("language") for s in scenarios if s.get("endpoint") == "assess"}
    assert {"zh-CN", "zh-TW", "en", "es", "pt"} <= languages
    # Output guarantees are pinned: source allow-list, origin labels, search cost, advice_source
    check_types = {c["type"] for s in scenarios for c in s["checks"]}
    assert {
        "sources_urls_allowed",
        "sources_origins",
        "search_count",
        "no_model_scores",
        "advice_source",
    } <= check_types


# ---------- Deliberate failure: how the failure library gets produced ----------


def test_failing_scenario_exits_1_and_dumps_failure(tmp_path, capsys):
    """A failing scenario: exit code 1, a dump holding request + response + failed checks,
    and output that carries the actual value.
    """
    path = write_scenarios(
        tmp_path,
        [
            {
                "id": "deliberate-fail",
                "description": "healthy form asserted as high risk on purpose",
                "endpoint": "assess",
                "request": full_form(),
                "checks": [
                    {"type": "status", "expect": 200},
                    {"type": "level", "model": "dengue", "expect_in": ["high"]},
                ],
            }
        ],
    )
    failures = tmp_path / "failures"
    assert main(["--scenarios", str(path), "--failures-dir", str(failures)]) == 1

    out = capsys.readouterr().out
    # The output must contain the actual value (a healthy form scores dengue as low)
    assert "got 'low'" in out
    assert "passed 0/1" in out

    dump_path = failures / "deliberate-fail.json"
    assert dump_path.is_file()
    dump = json.loads(dump_path.read_text(encoding="utf-8"))
    assert dump["id"] == "deliberate-fail"
    assert dump["request"]["age"] == 25
    assert dump["status"] == 200
    assert dump["response"]["dengue"]["level"] == "low"
    assert len(dump["failed_checks"]) == 1
    assert dump["failed_checks"][0]["type"] == "level"
    assert "low" in dump["failed_checks"][0]["detail"]


def test_stale_failure_dumps_cleared_between_runs(tmp_path):
    """Each run clears the previous run's failure dumps: the directory shows only the latest."""
    failures = tmp_path / "failures"
    failures.mkdir()
    stale = failures / "stale-scenario.json"
    stale.write_text("{}", encoding="utf-8")

    path = write_scenarios(
        tmp_path,
        [
            {
                "id": "all-good",
                "endpoint": "assess",
                "request": full_form(),
                "checks": [{"type": "status", "expect": 200}],
            }
        ],
    )
    assert main(["--scenarios", str(path), "--failures-dir", str(failures)]) == 0
    assert not stale.exists()
    assert list(failures.glob("*.json")) == []


# ---------- scores_match_scenario ----------


def test_scores_match_scenario_detects_mismatch(tmp_path, capsys):
    """Different symptoms always mean different z values: scores_match_scenario must go red."""
    path = write_scenarios(
        tmp_path,
        [
            {
                "id": "reference-febre",
                "endpoint": "assess",
                "request": full_form(yes_symptoms=("FEBRE",)),
                "checks": [{"type": "status", "expect": 200}],
            },
            {
                "id": "mismatched-twin",
                "endpoint": "assess",
                "request": full_form(),  # no fever, so z is bound to differ
                "checks": [
                    {"type": "status", "expect": 200},
                    {"type": "scores_match_scenario", "ref": "reference-febre"},
                ],
            },
        ],
    )
    failures = tmp_path / "failures"
    assert main(["--scenarios", str(path), "--failures-dir", str(failures)]) == 1

    out = capsys.readouterr().out
    assert "scores_match_scenario" in out
    assert "z" in out  # the difference detail must name the z value
    dump = json.loads(
        (failures / "mismatched-twin.json").read_text(encoding="utf-8")
    )
    assert any(c["type"] == "scores_match_scenario" for c in dump["failed_checks"])


def test_scores_match_scenario_resolves_ref_outside_only_filter(tmp_path):
    """When --only selects only the twin, the referenced baseline still runs on demand."""
    path = write_scenarios(
        tmp_path,
        [
            {
                "id": "base",
                "endpoint": "assess",
                "request": full_form(yes_symptoms=("FEBRE",)),
                "checks": [{"type": "status", "expect": 200}],
            },
            {
                "id": "twin",
                "endpoint": "assess",
                "request": full_form(yes_symptoms=("FEBRE",)),
                "checks": [{"type": "scores_match_scenario", "ref": "base"}],
            },
        ],
    )
    assert (
        main(
            [
                "--scenarios", str(path),
                "--failures-dir", str(tmp_path / "failures"),
                "--only", "twin",
            ]
        )
        == 0
    )


# ---------- New check types: sources_urls_allowed / advice_source ----------


def chat_request(question: str, **overrides) -> dict:
    body = {
        "language": "en",
        "question": question,
        "context": {"dengue": {"score": 42.2, "level": "medium"}},
        "history": [],
    }
    body.update(overrides)
    return body


def test_advice_source_check_detects_template_passed_off_as_llm(tmp_path, capsys):
    """Under MOCK the advice comes from a template; a scenario asserting llm must go red,
    with the actual value named in the detail.
    """
    path = write_scenarios(
        tmp_path,
        [
            {
                "id": "advice-source-mismatch",
                "endpoint": "assess",
                "request": full_form(),
                "checks": [{"type": "advice_source", "expect": "llm"}],
            }
        ],
    )
    assert main(["--scenarios", str(path), "--failures-dir", str(tmp_path / "f")]) == 1
    out = capsys.readouterr().out
    assert "advice_source" in out
    assert "'template'" in out


def test_sources_check_catches_unexpected_sources(tmp_path, capsys):
    """Mentioning Singapore triggers the tool, so asserting max_sources=0 must fail."""
    path = write_scenarios(
        tmp_path,
        [
            {
                "id": "unexpected-sources",
                "endpoint": "chat",
                "request": chat_request("I am going to Singapore, is dengue a risk?"),
                "checks": [{"type": "sources_urls_allowed", "max_sources": 0}],
            }
        ],
    )
    assert main(["--scenarios", str(path), "--failures-dir", str(tmp_path / "f")]) == 1
    out = capsys.readouterr().out
    assert "expected at most 0 source(s)" in out
    assert "who.int" in out  # the detail carries the actual link


def test_sources_check_catches_a_missing_source(tmp_path, capsys):
    """No place mentioned means no sources; asserting min_sources=1 must fail."""
    path = write_scenarios(
        tmp_path,
        [
            {
                "id": "missing-sources",
                "endpoint": "chat",
                "request": chat_request("What do my three scores mean?"),
                "checks": [{"type": "sources_urls_allowed", "min_sources": 1}],
            }
        ],
    )
    assert main(["--scenarios", str(path), "--failures-dir", str(tmp_path / "f")]) == 1
    assert "expected at least 1 source(s), got 0" in capsys.readouterr().out


# ---------- --only filter ----------


def test_only_filters_to_named_scenarios(tmp_path, capsys):
    failures = tmp_path / "failures"
    assert (
        main(
            [
                "--failures-dir", str(failures),
                "--only", "healthy-young-adult,chat-overlong-question",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "healthy-young-adult" in out
    assert "chat-overlong-question" in out
    assert "textbook-dengue" not in out
    assert "passed 2/2" in out


def test_only_with_unknown_id_is_a_usage_error(tmp_path, capsys):
    assert (
        main(["--failures-dir", str(tmp_path / "f"), "--only", "no-such-scenario"])
        == 2
    )
    assert "no-such-scenario" in capsys.readouterr().err


def test_missing_scenarios_file_is_a_usage_error(tmp_path, capsys):
    assert main(["--scenarios", str(tmp_path / "缺失.json")]) == 2
    assert "缺失" in capsys.readouterr().err


# ---------- --json output ----------


def test_json_output_parses_with_per_check_detail(tmp_path, capsys):
    failures = tmp_path / "failures"
    assert main(["--json", "--failures-dir", str(failures)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == payload["passed"] >= 15
    assert payload["failed"] == []
    assert len(payload["results"]) == payload["total"]
    for result in payload["results"]:
        assert set(result) >= {"id", "endpoint", "status", "passed", "checks"}
        assert result["passed"] is True
        assert len(result["checks"]) >= 1
        for check in result["checks"]:
            assert set(check) == {"type", "ok", "detail"}
            assert check["ok"] is True
            assert check["detail"]  # every check must have a readable detail


def test_json_output_reports_failure_detail(tmp_path, capsys):
    """In --json mode a failed check carries the actual-value detail as well."""
    path = write_scenarios(
        tmp_path,
        [
            {
                "id": "json-fail",
                "endpoint": "assess",
                "request": full_form(),
                "checks": [{"type": "exposure_level", "expect": "high"}],
            }
        ],
    )
    failures = tmp_path / "failures"
    assert main(["--json", "--scenarios", str(path), "--failures-dir", str(failures)]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["failed"] == ["json-fail"]
    (check,) = payload["results"][0]["checks"]
    assert check["ok"] is False
    assert "'low'" in check["detail"]  # the actual value
    assert payload["failure_dumps"] == [str(failures / "json-fail.json")]
