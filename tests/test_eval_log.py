"""评测数据回流（app/eval_log.py）与统计脚本（scripts/eval_stats.py）测试。

全部在 MOCK_MODE 下运行，不发真实网络请求；回流文件写到 pytest 临时目录。
"""

import json
from datetime import date, datetime

import pytest

# notes 中故意放入敏感标记文本，用于断言原文不落盘
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
    "language": "en",
    "notes": SENSITIVE_NOTE,
}


@pytest.fixture()
def log_path(tmp_path):
    return tmp_path / "assessments.jsonl"


def _make_client(monkeypatch, eval_log_path: str):
    """强制 MOCK_MODE=true 并指定回流路径后，构造 TestClient。"""
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


# ---------- 回流写入 ----------


def test_assess_appends_sanitized_record(client, log_path):
    resp = client.post("/api/assess", json=VALID_FORM)
    assert resp.status_code == 200
    body = resp.json()

    raw = log_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    # 三个模型的评分与 API 响应一致
    for field in ("dengue", "worsening", "severe"):
        assert record["scores"][field]["score"] == body[field]["score"]
        assert record["scores"][field]["level"] == body[field]["level"]
        assert record["scores"][field]["z"] == body[field]["z"]

    assert record["language"] == "en"
    assert record["mock_mode"] is True
    assert record["epi_week"] == body["epi_week"]

    # 26 个特征全部落盘，且与确定性编码一致
    from app.ml_model import encode_features
    from app.schemas import FEATS, FormInput

    expected = encode_features(FormInput(**VALID_FORM)).model_dump()
    assert record["features"] == expected
    assert set(record["features"]) == set(FEATS)

    # 时间戳为可解析的带时区 ISO 格式
    ts = datetime.fromisoformat(record["timestamp"])
    assert ts.tzinfo is not None

    # notes 原文绝不落盘，仅保留 has_notes 标记
    assert record["has_notes"] is True
    assert "notes" not in record
    assert SENSITIVE_NOTE not in raw
    assert "110101199001010011" not in raw


def test_multiple_assessments_append(client, log_path):
    for _ in range(3):
        assert client.post("/api/assess", json=VALID_FORM).status_code == 200
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)


def test_empty_path_disables_logging(monkeypatch, tmp_path):
    """EVAL_LOG_PATH 置空时关闭回流，默认相对路径也不会被创建。"""
    import app.eval_log as eval_log

    monkeypatch.setattr(eval_log, "_ROOT", tmp_path)
    with _make_client(monkeypatch, "") as c:
        assert c.post("/api/assess", json=VALID_FORM).status_code == 200

    from app.config import get_settings

    get_settings.cache_clear()
    assert not (tmp_path / "data" / "assessments.jsonl").exists()
    assert list(tmp_path.iterdir()) == []


def test_write_failure_does_not_break_assess(monkeypatch, tmp_path):
    """回流路径不可写（指向目录）时评估仍正常返回 200。"""
    with _make_client(monkeypatch, str(tmp_path)) as c:
        resp = c.post("/api/assess", json=VALID_FORM)

    from app.config import get_settings

    get_settings.cache_clear()
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["dengue"]["score"] <= 100.0


def test_relative_path_resolves_to_project_root(monkeypatch, tmp_path):
    """相对路径相对项目根目录（而非当前工作目录）解析。"""
    import app.eval_log as eval_log

    monkeypatch.setattr(eval_log, "_ROOT", tmp_path)
    with _make_client(monkeypatch, "data/assessments.jsonl") as c:
        assert c.post("/api/assess", json=VALID_FORM).status_code == 200

    from app.config import get_settings

    get_settings.cache_clear()
    assert (tmp_path / "data" / "assessments.jsonl").is_file()


# ---------- 统计脚本 ----------


def _record(
    dengue: float,
    severe: float,
    level: str,
    language: str = "zh-CN",
    mock: bool = True,
    week: int = 33,
) -> dict:
    def block(score: float) -> dict:
        return {"score": score, "level": level, "z": 0.0}

    return {
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
    # 直方图：100 分归入最后一档 [90-100]
    assert dengue["histogram"]["[10-20)"] == 1
    assert dengue["histogram"]["[90-100]"] == 1
    assert sum(dengue["histogram"].values()) == 4

    # 重症模型独立统计
    assert stats["models"]["severe"]["max"] == 95.0

    assert stats["languages"] == {"en": 1, "zh-CN": 3}
    assert stats["mock_count"] == 3
    assert stats["epi_weeks"] == {33: 4}


def test_compute_stats_empty():
    from scripts.eval_stats import compute_stats

    stats = compute_stats([])
    assert stats["total"] == 0
    assert stats["models"] == {}


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
    """回流记录里的 epi_week 与模型使用的一致。"""
    from app.ml_model import get_epi_week

    assert client.post("/api/assess", json=VALID_FORM).status_code == 200
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["epi_week"] == get_epi_week(date.today())
