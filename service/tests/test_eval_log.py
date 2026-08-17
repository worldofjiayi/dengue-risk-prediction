"""Tests for the evaluation feedback logging (app/eval_log.py) and the stats script
(scripts/eval_stats.py).

Everything runs under MOCK_MODE with no real network requests; the log file is written to
a pytest temporary directory.
"""

import json
from datetime import date, datetime

import pytest

# notes deliberately holds sensitive marker text, used to assert the raw text is not stored
SENSITIVE_NOTE = "最近发热，身份证号110101199001010011，请保密。"

VALID_FORM = {
    "age": 34,
    "sex": "F",
    "day_ill": 3,
    "symptoms": {
        "FEBRE": "yes",
        "CEFALEIA": "yes",
        "MIALGIA": "yes",
        "DOR_RETRO": "yes",
        "LEUCOPENIA": "unknown",
    },
    "comorbidities": {"DIABETES": "no", "HIPERTENSA": "yes"},
    "exposure": {"CONFIRMED_CASE": "yes", "FEVER_CLUSTER": "no"},
    "language": "en",
    "notes": SENSITIVE_NOTE,
}


@pytest.fixture()
def log_path(tmp_path):
    return tmp_path / "assessments.jsonl"


def _make_client(monkeypatch, eval_log_path: str):
    """Force MOCK_MODE=true, point the log at the given path, then build a TestClient."""
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("EVAL_LOG_PATH", eval_log_path)
    from app.config import get_settings

    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture()
def client(monkeypatch, log_path):
    with _make_client(monkeypatch, str(log_path)) as c:
        yield c

    from app.config import get_settings

    get_settings.cache_clear()


# ---------- Log writing ----------


def test_assess_appends_sanitized_record(client, log_path):
    resp = client.post("/api/assess", json=VALID_FORM)
    assert resp.status_code == 200
    body = resp.json()

    raw = log_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    # The three model scores agree with the API response
    for field in ("dengue", "worsening", "severe"):
        assert record["scores"][field]["score"] == body[field]["score"]
        assert record["scores"][field]["level"] == body[field]["level"]
        assert record["scores"][field]["z"] == body[field]["z"]

    assert record["language"] == "en"
    assert record["mock_mode"] is True
    assert record["epi_week"] == body["epi_week"]

    # All 26 features are stored, and match the deterministic encoding
    from app.ml_model import encode_features
    from app.schemas import FEATS, FormInput

    expected = encode_features(FormInput(**VALID_FORM)).model_dump()
    assert record["features"] == expected
    assert set(record["features"]) == set(FEATS)

    # The timestamp is a parseable ISO format carrying a timezone
    ts = datetime.fromisoformat(record["timestamp"])
    assert ts.tzinfo is not None

    # The raw notes text is never stored; only the has_notes flag is kept
    assert record["has_notes"] is True
    assert "notes" not in record
    assert SENSITIVE_NOTE not in raw
    assert "110101199001010011" not in raw


def test_record_includes_exposure_answers_and_level(client, log_path):
    """Exposure answers and the rule-derived level are both stored (categorical answers,
    which cannot identify an individual).
    """
    assert client.post("/api/assess", json=VALID_FORM).status_code == 200
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

    assert record["exposure"] == {
        "FEVER_CLUSTER": "no",
        "CONFIRMED_CASE": "yes",
        "OUTBREAK_TRAVEL": "unknown",  # keys left unanswered are filled in as unknown
    }
    assert record["exposure_level"] == "high"

    # Exposure answers must never leak into the 26 features
    from app.schemas import EXPOSURE_CODES, FEATS

    assert set(record["features"]) == set(FEATS)
    assert not set(record["features"]) & set(EXPOSURE_CODES)


def test_record_exposure_level_matches_response(client, log_path):
    """The logged exposure level agrees with the exposure_context the API returns."""
    body = client.post(
        "/api/assess",
        json={**VALID_FORM, "exposure": {"FEVER_CLUSTER": "yes"}},
    ).json()
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

    assert body["exposure_context"]["level"] == "medium"
    assert record["exposure_level"] == "medium"
    assert record["exposure"]["FEVER_CLUSTER"] == "yes"


def test_record_exposure_defaults_when_form_omits_it(client, log_path):
    form = {k: v for k, v in VALID_FORM.items() if k != "exposure"}
    assert client.post("/api/assess", json=form).status_code == 200
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

    assert record["exposure"] == {
        "FEVER_CLUSTER": "unknown",
        "CONFIRMED_CASE": "unknown",
        "OUTBREAK_TRAVEL": "unknown",
    }
    assert record["exposure_level"] == "low"


def test_multiple_assessments_append(client, log_path):
    for _ in range(3):
        assert client.post("/api/assess", json=VALID_FORM).status_code == 200
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)


def test_empty_path_disables_logging(monkeypatch, tmp_path):
    """An empty EVAL_LOG_PATH disables logging; the default relative path is not created."""
    import app.eval_log as eval_log

    monkeypatch.setattr(eval_log, "_ROOT", tmp_path)
    with _make_client(monkeypatch, "") as c:
        assert c.post("/api/assess", json=VALID_FORM).status_code == 200

    from app.config import get_settings

    get_settings.cache_clear()
    assert not (tmp_path / "data" / "assessments.jsonl").exists()
    assert list(tmp_path.iterdir()) == []


def test_write_failure_does_not_break_assess(monkeypatch, tmp_path):
    """When the log path is unwritable (it points at a directory), assess still returns 200."""
    with _make_client(monkeypatch, str(tmp_path)) as c:
        resp = c.post("/api/assess", json=VALID_FORM)

    from app.config import get_settings

    get_settings.cache_clear()
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["dengue"]["score"] <= 100.0


def test_relative_path_resolves_to_project_root(monkeypatch, tmp_path):
    """A relative path resolves against the project root, not the current working directory."""
    import app.eval_log as eval_log

    monkeypatch.setattr(eval_log, "_ROOT", tmp_path)
    with _make_client(monkeypatch, "data/assessments.jsonl") as c:
        assert c.post("/api/assess", json=VALID_FORM).status_code == 200

    from app.config import get_settings

    get_settings.cache_clear()
    assert (tmp_path / "data" / "assessments.jsonl").is_file()


# ---------- Stats script ----------


def _record(
    dengue: float,
    severe: float,
    level: str,
    language: str = "zh-CN",
    mock: bool = True,
    week: int = 33,
    exposure_level: str | None = "low",
) -> dict:
    def block(score: float) -> dict:
        return {"score": score, "level": level, "z": 0.0}

    record = {
        "timestamp": "2026-08-16T00:00:00+00:00",
        "language": language,
        "mock_mode": mock,
        "epi_week": week,
        "features": {},
        "scores": {
            "dengue": block(dengue),
            "worsening": block((dengue + severe) / 2),
            "severe": block(severe),
        },
        "has_notes": False,
    }
    # exposure_level=None simulates an old record from before the exposure questions existed
    if exposure_level is not None:
        record["exposure"] = {
            "FEVER_CLUSTER": "yes" if exposure_level == "medium" else "no",
            "CONFIRMED_CASE": "yes" if exposure_level == "high" else "no",
            "OUTBREAK_TRAVEL": "no",
        }
        record["exposure_level"] = exposure_level
    return record


def test_compute_stats():
    from scripts.eval_stats import compute_stats

    records = [
        _record(10.0, 5.0, "low"),
        _record(20.0, 15.0, "low", language="en"),
        _record(50.0, 45.0, "medium", mock=False),
        _record(100.0, 95.0, "high"),
    ]
    stats = compute_stats(records)

    assert stats["total"] == 4
    assert set(stats["models"]) == {"dengue", "worsening", "severe"}

    dengue = stats["models"]["dengue"]
    assert dengue["min"] == 10.0
    assert dengue["max"] == 100.0
    assert dengue["mean"] == 45.0
    assert dengue["median"] == 35.0
    assert dengue["levels"]["low"] == {"count": 2, "percent": 50.0}
    assert dengue["levels"]["high"] == {"count": 1, "percent": 25.0}
    # Histogram: a score of 100 lands in the last band [90-100]
    assert dengue["histogram"]["[10-20)"] == 1
    assert dengue["histogram"]["[90-100]"] == 1
    assert sum(dengue["histogram"].values()) == 4

    # The severe model is counted on its own
    assert stats["models"]["severe"]["max"] == 95.0

    assert stats["languages"] == {"en": 1, "zh-CN": 3}
    assert stats["mock_count"] == 3
    assert stats["epi_weeks"] == {33: 4}


def test_compute_stats_empty():
    from scripts.eval_stats import compute_stats

    stats = compute_stats([])
    assert stats["total"] == 0
    assert stats["models"] == {}
    assert stats["exposure_levels"] == {}


def test_compute_stats_exposure_level_distribution():
    from scripts.eval_stats import compute_stats

    records = [
        _record(10.0, 5.0, "low", exposure_level="low"),
        _record(20.0, 15.0, "low", exposure_level="low"),
        _record(50.0, 45.0, "medium", exposure_level="medium"),
        _record(90.0, 85.0, "high", exposure_level="high"),
    ]
    stats = compute_stats(records)
    assert stats["exposure_levels"] == {"high": 1, "low": 2, "medium": 1}


def test_compute_stats_ignores_records_without_exposure_level():
    """Old records from before the exposure questions must not be counted into a band."""
    from scripts.eval_stats import compute_stats

    records = [
        _record(10.0, 5.0, "low", exposure_level=None),   # old record
        _record(50.0, 45.0, "medium", exposure_level="high"),
    ]
    stats = compute_stats(records)
    assert stats["total"] == 2
    assert stats["exposure_levels"] == {"high": 1}


def test_eval_stats_report_prints_exposure_distribution(tmp_path, capsys):
    from scripts.eval_stats import compute_stats, print_report

    records = [
        _record(10.0, 5.0, "low", exposure_level="low"),
        _record(90.0, 85.0, "high", exposure_level="high"),
    ]
    print_report(compute_stats(records), 0, tmp_path / "assessments.jsonl")
    out = capsys.readouterr().out
    assert "流行病学暴露等级分布" in out
    assert "high" in out


def test_load_records_skips_malformed_lines(tmp_path):
    from scripts.eval_stats import load_records

    path = tmp_path / "assessments.jsonl"
    lines = [
        json.dumps(_record(30.0, 20.0, "low")),
        "这不是 JSON",
        json.dumps(["不是对象"]),
        json.dumps({"缺少": "scores 字段"}),
        "",
        json.dumps(_record(70.0, 60.0, "high")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records, skipped = load_records(path)
    assert len(records) == 2
    assert skipped == 3


def test_eval_stats_main_json_output(tmp_path, capsys):
    from scripts.eval_stats import main

    path = tmp_path / "assessments.jsonl"
    path.write_text(json.dumps(_record(80.0, 70.0, "high")) + "\n", encoding="utf-8")

    assert main([str(path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 1
    assert out["skipped"] == 0
    assert out["models"]["severe"]["levels"]["high"]["count"] == 1


def test_eval_stats_main_missing_file(tmp_path, capsys):
    from scripts.eval_stats import main

    assert main([str(tmp_path / "不存在.jsonl")]) == 1
    assert "文件不存在" in capsys.readouterr().err


def test_record_uses_real_epi_week(client, log_path):
    """The epi_week in the logged record is the same one the model uses."""
    from app.ml_model import get_epi_week

    assert client.post("/api/assess", json=VALID_FORM).status_code == 200
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["epi_week"] == get_epi_week(date.today())
