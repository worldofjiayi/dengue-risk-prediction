# pytest 根 conftest：确保项目根目录在 sys.path 中，使 tests 能 import app 包
import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _eval_log_to_tmp(monkeypatch, tmp_path):
    """测试默认把评测回流写到临时目录，避免污染项目 data/ 目录。

    单个测试可再用 monkeypatch.setenv("EVAL_LOG_PATH", ...) 覆盖。
    """
    monkeypatch.setenv("EVAL_LOG_PATH", str(tmp_path / "assessments.jsonl"))
