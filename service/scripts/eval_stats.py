"""评测数据统计：读取评估回流 JSONL，输出三个模型各自的评分分布与等级占比，
以及流行病学暴露等级的分布（规则判断，与模型评分相互独立）。

用法（Windows）：
    .venv\\Scripts\\python.exe scripts\\eval_stats.py                # 默认 data/assessments.jsonl
    .venv\\Scripts\\python.exe scripts\\eval_stats.py 路径\\xx.jsonl  # 指定文件
    .venv\\Scripts\\python.exe scripts\\eval_stats.py --json         # 机器可读 JSON（供导出）

每行一条记录，格式见 app/eval_log.py 的 build_record；坏行跳过并计数。

注意：模型阈值尚未在本地人群校准（见项目 README「已知局限」），
这份统计的用途正是为校准积累依据——重点看各模型评分的分布形态，
而不是把等级占比当作真实患病率。
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "assessments.jsonl"

# 评分直方图：0-100 分按 10 分一档，共 10 档（100 分归入最后一档）
BUCKET_WIDTH = 10
NUM_BUCKETS = 10

# 结果字段 -> 中文显示名
MODEL_FIELDS = {
    "dengue": "登革热可能性 (模型A)",
    "worsening": "病情加重风险 (模型B)",
    "severe": "重症风险 (模型B2)",
}


def load_records(path: Path) -> tuple[list[dict], int]:
    """读取 JSONL，返回 (合法记录列表, 跳过的坏行数)。

    坏行 = 非法 JSON、非对象、或缺少 scores 字段。
    """
    records: list[dict] = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get("scores"), dict):
                skipped += 1
                continue
            records.append(obj)
    return records, skipped


def _bucket_label(index: int) -> str:
    lo = index * BUCKET_WIDTH
    hi = lo + BUCKET_WIDTH
    return f"[{lo}-{hi})" if index < NUM_BUCKETS - 1 else f"[{lo}-{hi}]"


def _model_stats(records: list[dict], field: str) -> dict | None:
    """单个模型的评分统计；该字段完全缺失时返回 None。"""
    scores = [
        float(r["scores"][field]["score"])
        for r in records
        if isinstance(r["scores"].get(field), dict) and "score" in r["scores"][field]
    ]
    if not scores:
        return None

    histogram = {_bucket_label(i): 0 for i in range(NUM_BUCKETS)}
    for s in scores:
        index = min(int(s // BUCKET_WIDTH), NUM_BUCKETS - 1)
        histogram[_bucket_label(index)] += 1

    levels = Counter(
        str(r["scores"][field].get("level", "unknown"))
        for r in records
        if isinstance(r["scores"].get(field), dict)
    )
    total = sum(levels.values())
    return {
        "n": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": round(statistics.mean(scores), 2),
        "median": statistics.median(scores),
        "histogram": histogram,
        "levels": {
            level: {"count": c, "percent": round(c * 100.0 / total, 1)}
            for level, c in sorted(levels.items())
        },
    }


def compute_stats(records: list[dict]) -> dict:
    """从记录列表计算统计信息（纯函数，便于测试）。"""
    models = {}
    for field in MODEL_FIELDS:
        stats = _model_stats(records, field)
        if stats is not None:
            models[field] = stats

    # 流行病学暴露等级分布（规则判断，非模型输出）。
    # 只统计带该字段的记录：加入暴露问题之前的旧记录没有它，
    # 把它们计成 unknown 会凭空造出一个不存在的档位。
    exposure_levels = Counter(
        str(r["exposure_level"]) for r in records if "exposure_level" in r
    )

    return {
        "total": len(records),
        "models": models,
        "exposure_levels": dict(sorted(exposure_levels.items())),
        "languages": dict(
            sorted(Counter(str(r.get("language", "unknown")) for r in records).items())
        ),
        "epi_weeks": dict(
            sorted(Counter(int(r["epi_week"]) for r in records if "epi_week" in r).items())
        ),
        "mock_count": sum(1 for r in records if r.get("mock_mode")),
    }


def print_report(stats: dict, skipped: int, path: Path) -> None:
    """按人类可读格式打印统计报告。"""
    total = stats["total"]
    print(f"评测数据文件：{path}")
    print(f"记录总数：{total}（跳过坏行 {skipped} 条，其中 MOCK 记录 {stats['mock_count']} 条）")
    if total == 0:
        return

    for field, label in MODEL_FIELDS.items():
        m = stats["models"].get(field)
        if not m:
            continue
        print()
        print(f"===== {label} =====")
        print(
            f"n={m['n']}  min={m['min']}  max={m['max']}  "
            f"mean={m['mean']}  median={m['median']}"
        )
        max_count = max(m["histogram"].values()) or 1
        for lbl, count in m["histogram"].items():
            bar = "#" * round(count * 36 / max_count)
            print(f"  {lbl:>9}  {count:>5}  {bar}")
        levels = "  ".join(
            f"{lv}={info['count']}({info['percent']}%)"
            for lv, info in m["levels"].items()
        )
        print(f"  等级占比：{levels}")

    exposure = stats.get("exposure_levels") or {}
    if exposure:
        n = sum(exposure.values())
        print()
        print(f"流行病学暴露等级分布（规则判断，非模型输出；n={n}）：")
        for level in ("low", "medium", "high"):
            count = exposure.get(level, 0)
            print(f"  {level:>6}  {count:>5}  ({round(count * 100.0 / n, 1)}%)")
        for level, count in exposure.items():  # 兜底：出现意料之外的取值也要显示
            if level not in ("low", "medium", "high"):
                print(f"  {level:>6}  {count:>5}")

    print()
    print("语言分布：")
    for lang, count in stats["languages"].items():
        print(f"  {lang:>6}  {count:>5}")
    if stats["epi_weeks"]:
        weeks = "  ".join(f"W{w}={c}" for w, c in stats["epi_weeks"].items())
        print(f"\n流行病学周分布：{weeks}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估回流数据统计")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help="JSONL 文件路径（默认 data/assessments.jsonl）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出机器可读 JSON（含 skipped 字段），便于导出到其他工具",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.is_file():
        print(f"文件不存在：{path}", file=sys.stderr)
        return 1

    records, skipped = load_records(path)
    stats = compute_stats(records)
    if args.json:
        print(json.dumps({**stats, "skipped": skipped}, ensure_ascii=False, indent=2))
    else:
        print_report(stats, skipped, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
