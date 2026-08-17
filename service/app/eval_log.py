"""Evaluation feedback loop: after each assessment, append a de-identified record to a local JSONL file.

Its purpose is to accumulate local validation data for the dengue risk model (eval harness /
failure-case library) -- the project README's "known limitations" notes that the thresholds
have not been calibrated on the local population, and this logged data is the raw material
for that calibration.

One JSON object per line. The file holds **two kinds** of record, told apart by their fields
(not by their order):

  Assessment records (have scores): the 26 model features, score/level/z for all three
  models, the epidemiological week, a UTC timestamp, language, a mock_mode flag (so demo
  data can be filtered out during offline analysis), plus the three epidemiological
  exposure answers and the rule-derived exposure level.

  Search records (have search_count): one line for **every request that could possibly
  trigger a web search** in /api/chat and /api/destination, recording how many searches
  actually happened. Search is billed per call, and this is the only thing that can answer
  "what did this feature actually cost" after the fact.

Why the exposure answers may be written to disk: like the symptoms they are categorical
answers (yes/no/unknown) containing nothing that could identify a person, and they are
exactly the covariates we will most want when calibrating locally -- whether "a confirmed
case nearby" improves the model's discrimination can only be answered once enough data has
been collected. The raw notes text is never written to disk; only the has_notes boolean is.

A write failure is logged and nothing more; it never affects the main assessment flow.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.schemas import (
    EXPOSURE_CODES,
    ExposureContext,
    FormInput,
    MLFeatures,
    ModelScore,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

# Result field name -> model key
_MODEL_FIELDS = {"dengue": "A", "worsening": "B", "severe": "B2"}


def resolve_log_path(raw: str) -> Path:
    """Resolve relative paths against the project root, independent of uvicorn's cwd."""
    path = Path(raw)
    if not path.is_absolute():
        path = _ROOT / path
    return path


def build_record(
    form: FormInput,
    features: MLFeatures,
    scores: dict[str, ModelScore],
    epi_week: int,
    exposure: ExposureContext | None = None,
) -> dict:
    """Assemble one de-identified evaluation record (no raw notes or other sensitive fields).

    exposure is the rule-derived epidemiological exposure context; it is recorded together
    with the raw answers, but it **never** appears in features -- those 26 dimensions must
    match the training script exactly.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": form.language,
        "mock_mode": get_settings().mock_mode,
        "epi_week": epi_week,
        "features": features.model_dump(),
        "scores": {
            field: {
                "score": scores[key].score,
                "level": scores[key].level,
                "z": scores[key].z,
            }
            for field, key in _MODEL_FIELDS.items()
        },
        # Epidemiological exposure: not a model feature, kept in its own block so it
        # cannot be confused with features
        "exposure": {code: form.exposure.get(code, "unknown") for code in EXPOSURE_CODES},
        "exposure_level": exposure.level if exposure is not None else "low",
        "has_notes": bool(form.notes.strip()),
    }


def _append(record: dict, what: str) -> None:
    """Append one record to the log file; an empty EVAL_LOG_PATH turns logging off."""
    raw_path = get_settings().eval_log_path
    if not raw_path:
        return
    try:
        path = resolve_log_path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("%s写入失败（%s），本次请求结果不受影响", what, raw_path)


def log_assessment(
    form: FormInput,
    features: MLFeatures,
    scores: dict[str, ModelScore],
    epi_week: int,
    exposure: ExposureContext | None = None,
) -> None:
    """Append one evaluation record; an empty EVAL_LOG_PATH turns logging off."""
    _append(build_record(form, features, scores, epi_week, exposure), "评测记录")


def build_search_record(
    kind: str,
    language: str,
    location: str,
    search_count: int,
    search_status: str,
    matched: bool = False,
) -> dict:
    """Assemble one **search cost** record.

    Web search is the only thing in this service billed per call, and how much it costs is
    not up to us -- the model decides how many searches to run (measured: one ordinary
    question triggered 4). So every request that *could* search gets a line, including the
    ones that ended up not searching (search_count=0): logging only the ones that cost
    money would make "what share of requests actually cost anything" impossible to compute.

    location is the **normalised country/region name** (or the place name exactly as the
    user typed it); like the symptom answers it contains nothing that could identify a
    person, and the raw question text is never written to disk.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "language": language,
        "mock_mode": get_settings().mock_mode,
        "location": location,
        "matched": matched,
        "search_count": int(search_count),
        "search_status": search_status,
    }


def log_search(
    kind: str,
    language: str,
    location: str,
    search_count: int,
    search_status: str,
    matched: bool = False,
) -> None:
    """Append one search cost record; a write failure is only logged, never affecting the response."""
    _append(
        build_search_record(
            kind, language, location, search_count, search_status, matched
        ),
        "检索记录",
    )
