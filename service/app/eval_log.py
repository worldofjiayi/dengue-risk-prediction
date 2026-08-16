"""评测数据回流：每次评估完成后，把脱敏记录追加写入本地 JSONL 文件。

用于给登革热风险模型积累本地验证数据（eval harness / 失败案例库）——
项目 README 的「已知局限」指出阈值未在本地人群校准，这份回流数据就是校准的原料。

每行一条 JSON：26 个模型特征、三个模型的 score/level/z、流行病学周、
UTC 时间戳、language、mock_mode 标记（供离线分析时过滤演示数据）。
notes 原文绝不落盘，仅记录 has_notes 布尔值。

写入失败只记日志，绝不影响评估主流程。
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.schemas import FormInput, MLFeatures, ModelScore

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

# 结果字段名 -> 模型键
_MODEL_FIELDS = {"dengue": "A", "worsening": "B", "severe": "B2"}


def resolve_log_path(raw: str) -> Path:
    """相对路径相对项目根目录解析，保证与 uvicorn 启动目录无关。"""
    path = Path(raw)
    if not path.is_absolute():
        path = _ROOT / path
    return path


def build_record(
    form: FormInput,
    features: MLFeatures,
    scores: dict[str, ModelScore],
    epi_week: int,
) -> dict:
    """组装一条脱敏评测记录（不含 notes 原文等敏感字段）。"""
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
        "has_notes": bool(form.notes.strip()),
    }


def log_assessment(
    form: FormInput,
    features: MLFeatures,
    scores: dict[str, ModelScore],
    epi_week: int,
) -> None:
    """追加一条评测记录；EVAL_LOG_PATH 为空时关闭回流。"""
    raw_path = get_settings().eval_log_path
    if not raw_path:
        return
    record = build_record(form, features, scores, epi_week)
    try:
        path = resolve_log_path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("评测记录写入失败（%s），本次评估结果不受影响", raw_path)
