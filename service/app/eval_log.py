"""评测数据回流：每次评估完成后，把脱敏记录追加写入本地 JSONL 文件。

用于给登革热风险模型积累本地验证数据（eval harness / 失败案例库）——
项目 README 的「已知局限」指出阈值未在本地人群校准，这份回流数据就是校准的原料。

每行一条 JSON。文件里有**两种**记录，靠字段区分（不是靠顺序）：

  评估记录（有 scores）：26 个模型特征、三个模型的 score/level/z、流行病学周、
  UTC 时间戳、language、mock_mode 标记（供离线分析时过滤演示数据），
  以及三项流行病学暴露答案与规则判出的暴露等级。

  检索记录（有 search_count）：/api/chat 与 /api/destination 里**每一次
  有可能联网检索的请求**各一行，记下真的检索了几次。检索按次计费，
  这是唯一能事后回答「这个功能到底花了多少钱」的东西。

暴露答案之所以可以落盘：它们和症状一样是分类答案（yes/no/unknown），
不含任何可定位到个人的信息，而且正是将来做本地校准时最想知道的协变量——
「身边有确诊病例」能不能提升模型区分度，只有攒够数据才答得上来。
notes 原文绝不落盘，仅记录 has_notes 布尔值。

写入失败只记日志，绝不影响评估主流程。
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
    exposure: ExposureContext | None = None,
) -> dict:
    """组装一条脱敏评测记录（不含 notes 原文等敏感字段）。

    exposure 是规则判出的流行病学暴露背景；连同原始答案一起记录，
    但**不会**出现在 features 里——那 26 维必须与训练脚本严格一致。
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
        # 流行病学暴露：非模型特征，单独成块，避免与 features 混淆
        "exposure": {code: form.exposure.get(code, "unknown") for code in EXPOSURE_CODES},
        "exposure_level": exposure.level if exposure is not None else "low",
        "has_notes": bool(form.notes.strip()),
    }


def _append(record: dict, what: str) -> None:
    """把一条记录追加进回流文件；EVAL_LOG_PATH 为空时关闭回流。"""
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
    """追加一条评测记录；EVAL_LOG_PATH 为空时关闭回流。"""
    _append(build_record(form, features, scores, epi_week, exposure), "评测记录")


def build_search_record(
    kind: str,
    language: str,
    location: str,
    search_count: int,
    search_status: str,
    matched: bool = False,
) -> dict:
    """组装一条**检索花销**记录。

    联网检索是这个服务里唯一按次计费的东西，而且花多少不由我们决定——
    模型自己决定检索几次（实测一个普通问题触发了 4 次）。因此每一次
    *有可能*检索的请求都要记一行，包括最后没检索的（search_count=0）：
    只记花了钱的那些，就永远算不出「多少比例的请求真的花了钱」。

    location 是**规范化后的国家/地区名**（或用户原样输入的地名），
    与症状答案一样不含可定位到个人的信息；问题原文绝不落盘。
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
    """追加一条检索花销记录；写入失败只记日志，绝不影响响应。"""
    _append(
        build_search_record(
            kind, language, location, search_count, search_status, matched
        ),
        "检索记录",
    )
