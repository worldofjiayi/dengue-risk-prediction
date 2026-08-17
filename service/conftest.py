# Root pytest conftest: keep the project root on sys.path so tests can import the app package
import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _eval_log_to_tmp(monkeypatch, tmp_path):
    """By default, send the evaluation feedback log to a temp directory during tests,
    so the project's data/ directory is not polluted.

    An individual test can still override this with monkeypatch.setenv("EVAL_LOG_PATH", ...).
    """
    monkeypatch.setenv("EVAL_LOG_PATH", str(tmp_path / "assessments.jsonl"))
