"""场景化评测运行器（scripts/eval_run.py）测试。

元测试思路：harness 本身也是被测对象——
  1. 随包的 scenarios.json 必须全绿（这就是回归门禁的基线）；
  2. 故意失败的场景必须产生退出码 1、失败转储文件、以及带实际值的输出；
  3. scores_match_scenario 必须能识破 z 值不一致；
  4. --only 过滤与 --json 输出必须可用。

全部在 MOCK_MODE 下进程内运行，不发真实网络请求。
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
    """runner 会直接改 os.environ；先用 monkeypatch 设一遍，让 teardown 能还原。"""
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("EVAL_LOG_PATH", "")


def full_form(yes_symptoms=(), **overrides) -> dict:
    """构造一份键齐全的问卷请求体。"""
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


# ---------- 随包场景：回归门禁基线 ----------


def test_shipped_scenarios_all_pass(tmp_path, capsys):
    """元测试：随包的 scenarios.json 必须全部通过（退出码 0，无失败转储）。"""
    failures = tmp_path / "failures"
    assert main(["--failures-dir", str(failures)]) == 0

    out = capsys.readouterr().out
    assert "passed" in out
    assert "failure library" not in out
    # 全绿时失败目录存在但为空
    assert failures.is_dir()
    assert list(failures.glob("*.json")) == []


def test_shipped_scenarios_file_is_valid_and_broad():
    """场景文件本身合法，且覆盖面达到基线：两类端点、约 15-20 个场景。"""
    scenarios = load_scenarios(DEFAULT_SCENARIOS)
    assert 15 <= len(scenarios) <= 20
    endpoints = {s.get("endpoint", "assess") for s in scenarios}
    assert endpoints == {"assess", "chat"}
    # 五种语言都必须被 assess 场景覆盖
    languages = {s["request"].get("language") for s in scenarios if s.get("endpoint") == "assess"}
    assert {"zh-CN", "zh-TW", "en", "es", "pt"} <= languages


# ---------- 故意失败：失败案例库的产生路径 ----------


def test_failing_scenario_exits_1_and_dumps_failure(tmp_path, capsys):
    """失败场景：退出码 1，落盘转储含请求+响应+失败检查，输出带实际值。"""
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
    # 输出必须包含实际值（healthy 问卷的 dengue 等级是 low）
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
    """每轮运行开始时清空上一轮的失败转储：目录只反映最新一轮。"""
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
    """症状不同的两份问卷 z 值必然不同：scores_match_scenario 必须报红。"""
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
                "request": full_form(),  # 无发热，z 必然不同
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
    assert "z" in out  # 差异明细里必须点名 z 值
    dump = json.loads(
        (failures / "mismatched-twin.json").read_text(encoding="utf-8")
    )
    assert any(c["type"] == "scores_match_scenario" for c in dump["failed_checks"])


def test_scores_match_scenario_resolves_ref_outside_only_filter(tmp_path):
    """--only 只选了对照场景时，被引用的基准场景仍能按需执行。"""
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


# ---------- --only 过滤 ----------


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


# ---------- --json 输出 ----------


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
            assert check["detail"]  # 每条检查都要有可读明细


def test_json_output_reports_failure_detail(tmp_path, capsys):
    """--json 模式下失败检查同样带实际值明细。"""
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
    assert "'low'" in check["detail"]  # 实际值
    assert payload["failure_dumps"] == [str(failures / "json-fail.json")]
