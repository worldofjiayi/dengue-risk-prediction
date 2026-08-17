"""Evaluation data statistics: read the assessment feedback JSONL and report the score
distribution and level shares of each of the three models, the distribution of
epidemiological exposure levels (a rule-based judgement, independent of the model
scores), and **what web search costs**.

The feedback file holds two kinds of record (see app/eval_log.py): assessment records
carrying scores, and search records carrying search_count. The two are counted
separately -- search records have no scores and assessment records do not search, so
throwing them into a single denominator would only produce a meaningless number.

Usage (Windows):
    .venv\\Scripts\\python.exe scripts\\eval_stats.py                 # default data/assessments.jsonl
    .venv\\Scripts\\python.exe scripts\\eval_stats.py path\\xx.jsonl  # a specific file
    .venv\\Scripts\\python.exe scripts\\eval_stats.py --json          # machine-readable JSON (export)

One record per line, format per build_record in app/eval_log.py; bad lines are skipped
and counted.

Note: the model thresholds have not yet been calibrated on the local population (see
"Known limitations" in the project README). The purpose of these statistics is precisely
to accumulate evidence for that calibration -- look at the shape of each model's score
distribution, rather than treating the level shares as a real prevalence.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "assessments.jsonl"

# Score histogram: 0-100 in bands of 10, 10 bands in all (a score of 100 goes in the last)
BUCKET_WIDTH = 10
NUM_BUCKETS = 10

# Result field -> display name
MODEL_FIELDS = {
    "dengue": "Dengue likelihood (model A)",
    "worsening": "Worsening risk (model B)",
    "severe": "Severe risk (model B2)",
}


def is_assessment(record: dict) -> bool:
    """Assessment record: carries a scores object."""
    return isinstance(record.get("scores"), dict)


def is_search(record: dict) -> bool:
    """Search record: carries an integer search_count (bool subclasses int, so block it)."""
    count = record.get("search_count")
    return isinstance(count, int) and not isinstance(count, bool)


def load_records(path: Path) -> tuple[list[dict], int]:
    """Read the JSONL and return (list of valid records, number of bad lines skipped).

    A bad line = invalid JSON, not an object, or looking like neither kind of record
    (no scores and no search_count).
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
    """Score statistics for one model; returns None when the field is missing entirely."""
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
    """Cost statistics for web search.

    The denominator is **every request that could have searched** (including those that
    ended at search_count=0): counting only the ones that cost money would never let you
    work out "what fraction of requests really cost anything", and that fraction is the
    only evidence that the rule "search only when a location is recognised" is working.
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
    """Compute the statistics from a list of records (a pure function, easy to test).

    Assessment and search records are counted separately: total / models / languages and
    the rest look only at assessment records, and the search block only at search
    records. A denominator that mixes the two means nothing at all.
    """
    records = [r for r in all_records if is_assessment(r)]
    search_records = [r for r in all_records if is_search(r) and not is_assessment(r)]

    models = {}
    for field in MODEL_FIELDS:
        stats = _model_stats(records, field)
        if stats is not None:
            models[field] = stats

    # Distribution of epidemiological exposure levels (a rule-based judgement, not model
    # output). Only records that carry the field are counted: older records from before
    # the exposure questions were added do not have it, and counting them as unknown
    # would conjure up a band that does not exist.
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
    """Web search cost: total calls, the mean, and the share of requests with no search."""
    if not search.get("n"):
        return
    n = search["n"]
    print()
    print(f"Web search cost (requests that could search, n={n}):")
    print(
        f"  {search['total']} search(es) total  mean {search['mean']} per request  "
        f"max {search['max']} in one request  {search['zero']} with no search"
        f" ({round(search['zero'] * 100.0 / n, 1)}%)"
    )
    for kind, bucket in search.get("by_kind", {}).items():
        print(f"  {kind:>12}  n={bucket['n']:<5} total {bucket['total']:<5} mean {bucket['mean']}")
    statuses = "  ".join(f"{k}={v}" for k, v in search.get("statuses", {}).items())
    if statuses:
        print(f"  Status distribution: {statuses}")


def print_report(stats: dict, skipped: int, path: Path) -> None:
    """Print the statistics report in a human-readable format."""
    total = stats["total"]
    search = stats.get("search") or {}
    print(f"Evaluation data file: {path}")
    print(
        f"Records: {total} ({skipped} malformed line(s) skipped, "
        f"{stats['mock_count']} MOCK record(s))"
    )
    if total == 0:
        _print_search(search)  # a file with only search records should still show its cost
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
        print(f"  Level shares: {levels}")

    exposure = stats.get("exposure_levels") or {}
    if exposure:
        n = sum(exposure.values())
        print()
        print(
            f"Epidemiological exposure level distribution "
            f"(rule-based, not model output; n={n}):"
        )
        for level in ("low", "medium", "high"):
            count = exposure.get(level, 0)
            print(f"  {level:>6}  {count:>5}  ({round(count * 100.0 / n, 1)}%)")
        for level, count in exposure.items():  # fallback: unexpected values must show too
            if level not in ("low", "medium", "high"):
                print(f"  {level:>6}  {count:>5}")

    print()
    print("Language distribution:")
    for lang, count in stats["languages"].items():
        print(f"  {lang:>6}  {count:>5}")
    if stats["epi_weeks"]:
        weeks = "  ".join(f"W{w}={c}" for w, c in stats["epi_weeks"].items())
        print(f"\nEpidemiological week distribution: {weeks}")

    _print_search(search)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Statistics over the evaluation log data")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help="path to the JSONL file (default data/assessments.jsonl)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON (including the skipped field) for export to other tools",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.is_file():
        print(f"File does not exist: {path}", file=sys.stderr)
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
