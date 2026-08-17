"""评测数据统计：读取评估回流 JSONL，输出三个模型各自的评分分布与等级占比、
流行病学暴露等级的分布（规则判断，与模型评分相互独立），以及**联网检索的花销**。

回流文件里有两种记录（见 app/eval_log.py）：带 scores 的评估记录，
带 search_count 的检索记录。两者分开统计——检索记录没有评分，评估记录不检索，
把它们混进同一个分母只会得到没有意义的数字。

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


def is_assessment(record: dict) -> bool:
    """评估记录：带 scores 对象。"""
    return isinstance(record.get("scores"), dict)


def is_search(record: dict) -> bool:
    """检索记录：带整数 search_count（bool 是 int 的子类，要挡掉）。"""
    count = record.get("search_count")
    return isinstance(count, int) and not isinstance(count, bool)


def load_records(path: Path) -> tuple[list[dict], int]:
    """读取 JSONL，返回 (合法记录列表, 跳过的坏行数)。

    坏行 = 非法 JSON、非对象、或两种记录都不像（既没有 scores 也没有 search_count）。
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
            if not isinstance(obj, dict) or not (is_assessment(obj) or is_search(obj)):
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


def _search_stats(records: list[dict]) -> dict:
    """联网检索的花销统计。

    分母是**所有可能检索的请求**（包括最终 search_count=0 的那些）：
    只统计花了钱的那些，就永远算不出「多少比例的请求真的花了钱」，
    而这正是「只在识别到地点时才检索」这条规则有没有生效的唯一证据。
    """
    counts = [int(r["search_count"]) for r in records]
    if not counts:
        return {
            "n": 0,
            "total": 0,
            "mean": 0.0,
            "max": 0,
            "zero": 0,
            "by_kind": {},
            "statuses": {},
        }
    by_kind: dict[str, dict] = {}
    for record in records:
        kind = str(record.get("kind", "unknown"))
        bucket = by_kind.setdefault(kind, {"n": 0, "total": 0})
        bucket["n"] += 1
        bucket["total"] += int(record["search_count"])
    for bucket in by_kind.values():
        bucket["mean"] = round(bucket["total"] / bucket["n"], 2)
    return {
        "n": len(counts),
        "total": sum(counts),
        "mean": round(statistics.mean(counts), 2),
        "max": max(counts),
        "zero": sum(1 for c in counts if c == 0),
        "by_kind": dict(sorted(by_kind.items())),
        "statuses": dict(
            sorted(Counter(str(r.get("search_status", "unknown")) for r in records).items())
        ),
    }


def compute_stats(all_records: list[dict]) -> dict:
    """从记录列表计算统计信息（纯函数，便于测试）。

    评估记录与检索记录分开统计：total / models / languages 等只看评估记录，
    search 块只看检索记录。混在一起的分母没有任何意义。
    """
    records = [r for r in all_records if is_assessment(r)]
    search_records = [r for r in all_records if is_search(r) and not is_assessment(r)]

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
        "search": _search_stats(search_records),
        "exposure_levels": dict(sorted(exposure_levels.items())),
        "languages": dict(
            sorted(Counter(str(r.get("language", "unknown")) for r in records).items())
        ),
        "epi_weeks": dict(
            sorted(Counter(int(r["epi_week"]) for r in records if "epi_week" in r).items())
        ),
        "mock_count": sum(1 for r in records if r.get("mock_mode")),
    }


def _print_search(search: dict) -> None:
    """联网检索花销：总次数、均值，以及零检索请求的占比。"""
    if not search.get("n"):
        return
    n = search["n"]
    print()
    print(f"联网检索花销（可能检索的请求 n={n}）：")
    print(
        f"  总检索次数 {search['total']}  均值 {search['mean']} 次/请求  "
        f"单次最多 {search['max']} 次  零检索 {search['zero']} 次"
        f"（{round(search['zero'] * 100.0 / n, 1)}%）"
    )
    for kind, bucket in search.get("by_kind", {}).items():
        print(f"  {kind:>12}  n={bucket['n']:<5} 合计 {bucket['total']:<5} 均值 {bucket['mean']}")
    statuses = "  ".join(f"{k}={v}" for k, v in search.get("statuses", {}).items())
    if statuses:
        print(f"  状态分布：{statuses}")


def print_report(stats: dict, skipped: int, path: Path) -> None:
    """按人类可读格式打印统计报告。"""
    total = stats["total"]
    search = stats.get("search") or {}
    print(f"评测数据文件：{path}")
    print(f"记录总数：{total}（跳过坏行 {skipped} 条，其中 MOCK 记录 {stats['mock_count']} 条）")
    if total == 0:
        _print_search(search)  # 只有检索记录的文件也该看得到花销
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

    _print_search(search)


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
