"""场景化评测运行器（eval harness）：回归门禁 + 失败案例库。

对 service/eval/scenarios.json 里的每个场景，用 FastAPI TestClient 在**进程内**
调用 /api/assess 或 /api/chat（强制 MOCK_MODE=true，不发任何真实网络请求），
逐条执行声明式检查（check），最后汇总通过/失败。

用法（Windows，项目根目录）：
    .venv\\Scripts\\python.exe service\\scripts\\eval_run.py
    .venv\\Scripts\\python.exe service\\scripts\\eval_run.py --json
    .venv\\Scripts\\python.exe service\\scripts\\eval_run.py --only healthy-young-adult,textbook-dengue
    .venv\\Scripts\\python.exe service\\scripts\\eval_run.py --scenarios 路径\\其他场景.json

失败的场景会把请求+响应完整落盘到 service/eval/failures/<id>.json（失败案例库），
每次运行前清空上一次的残留文件。退出码：0 全部通过，1 有失败，2 用法/文件错误。

这份 harness 的定位（项目方法论）：它是「唯一能告诉你要不要重训模型」的东西——
场景固化的是**当前已验证的行为**，模型或规则一旦变化，先在这里看见。
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIOS = SERVICE_ROOT / "eval" / "scenarios.json"
DEFAULT_FAILURES_DIR = SERVICE_ROOT / "eval" / "failures"

ENDPOINTS = {"assess": "/api/assess", "chat": "/api/chat"}

# 响应里三个模型的字段名（与 AssessmentResult 一致）
MODEL_FIELDS = ("dengue", "worsening", "severe")

# advice 对象的键必须按此顺序序列化（就医优先，见 schemas.Advice 的说明）
ADVICE_ORDER = ["medical", "monitoring", "protection"]

LEVEL_ORDER = ("low", "medium", "high")

# scores_match_scenario 的 z 值比较容差：同一进程、同一系数、同一日期下
# 两次评估的 z 应当逐位相同，容差只为吸收浮点噪声。
Z_TOLERANCE = 1e-6

# ---- 各语言的「就医/急迫性」关键词表（medical_urgency 检查） ----
# 关键词固化的是当前 MOCK 建议文案里确实存在的表述；真实 DeepSeek 输出
# 也应命中同一词表——命不中就是值得进失败案例库的输出。
URGENCY_KEYWORDS: dict[str, list[str]] = {
    "zh-CN": ["就医", "就诊", "急诊"],
    "zh-TW": ["就醫", "就診", "急診"],
    "en": [
        "seek medical",
        "seek care",
        "medical care",
        "see a clinician",
        "medical review",
        "emergency department",
    ],
    "es": ["atención médica", "acuda", "consulte", "consulta médica"],
    "pt": [
        "procure atendimento",
        "atendimento médico",
        "procure um profissional",
        "avaliação médica",
        "pronto-socorro",
    ],
}

# ---- 追问回复必须提到的总体档位标签（reply_mentions_tier 检查） ----
# 与 app/deepseek_client.py 的 _MOCK_CHAT_TIER_LABELS 一致；刻意**不**从
# app 包导入——harness 固化的是期望行为，文案变了应当在这里报红，而不是静默跟随。
TIER_LABELS: dict[str, dict[str, str]] = {
    "zh-CN": {"low": "较低", "medium": "中等", "high": "偏高"},
    "zh-TW": {"low": "較低", "medium": "中等", "high": "偏高"},
    "en": {"low": "low", "medium": "moderate", "high": "high"},
    "es": {"low": "bajo", "medium": "moderado", "high": "alto"},
    "pt": {"low": "baixo", "medium": "moderado", "high": "alto"},
}

# ---- 概率化表述检测（no_probability_language 检查） ----
# 同一字符串里同时出现「数字百分比」与「概率类措辞」即判失败：
# 模型是无截距/相对评分，任何 "37% 概率感染" 式的输出都是违约文案。
PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
PROBABILITY_RE = re.compile(
    r"概率|几率|機率|probabilit|probabilidad|probabilidade|chance|likelihood",
    re.IGNORECASE,
)

# ---- 引用链接检测（sources_urls_allowed 检查） ----
# 同样刻意不从 app.verifier 导入：harness 用**独立写的**正则与更严格的
# 「精确相等」判定去核对同一条不变量——回复里出现的每个链接都必须真的在
# 本轮 sources 里。两边都能通过，才说明这条不变量不是靠某一处实现巧合成立的。
URL_RE = re.compile(r"https?://[^\s<>\"'）)】\[\]（(，。；、]+")
URL_TRAILING = ".,;:!?'\")]}>，。；！？、）】"

ADVICE_SOURCES = ("llm", "template")

# 中文（含繁体）语言代码：关键词/标签用子串匹配，拉丁语言用词边界匹配
_CJK_LANGUAGES = ("zh-CN", "zh-TW")


# ---------- 基础设施 ----------


def build_client():
    """强制 MOCK_MODE、关闭评测回流后，构造进程内 TestClient。

    与 tests/test_pipeline.py 的 client fixture 同一套路：先设环境变量、
    清掉配置缓存，再导入 app。EVAL_LOG_PATH 置空是为了不让 harness 流量
    混进 data/assessments.jsonl 的真实回流数据。
    """
    os.environ["MOCK_MODE"] = "true"
    os.environ["EVAL_LOG_PATH"] = ""
    if str(SERVICE_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVICE_ROOT))

    # 压掉 app / httpx 的 INFO 日志：18 个场景的流水线日志会把结果行淹掉
    for name in ("app", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)

    from app.config import get_settings

    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def load_scenarios(path: Path) -> list[dict]:
    """读取并粗校验场景文件：必须是对象列表，id 唯一，endpoint 合法。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"场景文件顶层必须是 JSON 列表：{path}")
    seen: set[str] = set()
    for scenario in raw:
        if not isinstance(scenario, dict) or "id" not in scenario:
            raise ValueError("每个场景必须是含 id 的 JSON 对象")
        sid = scenario["id"]
        if sid in seen:
            raise ValueError(f"场景 id 重复：{sid}")
        seen.add(sid)
        endpoint = scenario.get("endpoint", "assess")
        if endpoint not in ENDPOINTS:
            raise ValueError(f"场景 {sid} 的 endpoint 非法：{endpoint}")
        if "request" not in scenario or "checks" not in scenario:
            raise ValueError(f"场景 {sid} 缺少 request 或 checks")
    return raw


def _request_language(scenario: dict) -> str:
    return scenario.get("request", {}).get("language", "zh-CN")


def _dig(body, path: str):
    """按点号路径取值，如 "dengue.z"、"advice.medical.0"。列表段用整数下标。"""
    cur = body
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(path)
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def _context_tier(scenario: dict) -> str:
    """追问场景：从前端回传的 context 里取三模型等级的最高档（同 pipeline.overall_tier）。"""
    context = scenario.get("request", {}).get("context") or {}
    best = "low"
    for field in MODEL_FIELDS:
        block = context.get(field)
        if isinstance(block, dict):
            level = block.get("level")
            if level in LEVEL_ORDER and LEVEL_ORDER.index(level) > LEVEL_ORDER.index(best):
                best = level
    return best


def _gather_text_fields(body: dict) -> list[str]:
    """no_probability_language 检查的扫描范围：summary + 三类建议的每一条。"""
    texts: list[str] = []
    summary = body.get("summary")
    if isinstance(summary, str):
        texts.append(summary)
    advice = body.get("advice")
    if isinstance(advice, dict):
        for items in advice.values():
            if isinstance(items, list):
                texts.extend(s for s in items if isinstance(s, str))
    return texts


# ---------- 检查实现 ----------
# 每个检查函数：(check, scenario, status, body, runner) -> (ok, detail)
# detail 必须带上**实际值**——失败时能直接看出差在哪，这是失败案例库的意义。


def _check_status(check, scenario, status, body, runner):
    expect = check.get("expect")
    return status == expect, f"expected HTTP {expect}, got {status}"


def _check_level(check, scenario, status, body, runner):
    model = check.get("model")
    expect_in = check.get("expect_in", [])
    if not isinstance(body, dict) or not isinstance(body.get(model), dict):
        return False, f"response has no model block '{model}' (HTTP {status})"
    block = body[model]
    got = block.get("level")
    ok = got in expect_in
    return ok, f"{model}.level expected in {expect_in}, got '{got}' (score={block.get('score')})"


def _check_score_between(check, scenario, status, body, runner):
    model = check.get("model")
    lo, hi = check.get("lo"), check.get("hi")
    if not isinstance(body, dict) or not isinstance(body.get(model), dict):
        return False, f"response has no model block '{model}' (HTTP {status})"
    got = body[model].get("score")
    ok = isinstance(got, (int, float)) and lo <= got <= hi
    return ok, f"{model}.score expected in [{lo}, {hi}], got {got}"


def _check_warning_signs(check, scenario, status, body, runner):
    expect = check.get("expect", [])
    if not isinstance(body, dict) or not isinstance(body.get("warning_signs"), list):
        return False, f"response has no warning_signs list (HTTP {status})"
    got = body["warning_signs"]
    ok = set(got) == set(expect)
    return ok, f"warning_signs expected {sorted(expect)}, got {sorted(got)}"


def _check_exposure_level(check, scenario, status, body, runner):
    expect = check.get("expect")
    try:
        got = _dig(body, "exposure_context.level")
    except (KeyError, TypeError, ValueError, IndexError):
        return False, f"response has no exposure_context.level (HTTP {status})"
    return got == expect, f"exposure_context.level expected '{expect}', got '{got}'"


def _check_field_equals(check, scenario, status, body, runner):
    path, expect = check.get("path"), check.get("expect")
    try:
        got = _dig(body, path)
    except (KeyError, TypeError, ValueError, IndexError):
        return False, f"path '{path}' not found in response (HTTP {status})"
    return got == expect, f"{path} expected {expect!r}, got {got!r}"


def _check_advice_order(check, scenario, status, body, runner):
    if not isinstance(body, dict) or not isinstance(body.get("advice"), dict):
        return False, f"response has no advice object (HTTP {status})"
    got = list(body["advice"].keys())
    return got == ADVICE_ORDER, f"advice keys expected {ADVICE_ORDER}, got {got}"


def _check_medical_urgency(check, scenario, status, body, runner):
    language = _request_language(scenario)
    keywords = URGENCY_KEYWORDS.get(language)
    if keywords is None:
        return False, f"no urgency lexicon for language '{language}'"
    try:
        items = _dig(body, "advice.medical")
    except (KeyError, TypeError, ValueError, IndexError):
        return False, f"response has no advice.medical (HTTP {status})"
    if not isinstance(items, list) or not items:
        return False, f"advice.medical is empty or not a list: {items!r}"
    for item in items:
        text = str(item).lower()
        if any(k.lower() in text for k in keywords):
            return True, f"matched urgency keyword in advice.medical ({language})"
    return False, (
        f"no advice.medical item matched urgency lexicon {keywords} "
        f"(language={language}); items={items!r}"
    )


def _check_no_probability_language(check, scenario, status, body, runner):
    if not isinstance(body, dict):
        return False, f"response body is not an object (HTTP {status})"
    offending = [
        text
        for text in _gather_text_fields(body)
        if PERCENT_RE.search(text) and PROBABILITY_RE.search(text)
    ]
    if offending:
        return False, f"probability-claim wording found in advice/summary: {offending!r}"
    return True, "no probability-claim wording in advice/summary"


def _check_explanations_present(check, scenario, status, body, runner):
    if not isinstance(body, dict) or not isinstance(body.get("explanations"), dict):
        return False, f"response has no explanations object (HTTP {status})"
    explanations = body["explanations"]
    missing = set(MODEL_FIELDS) - set(explanations)
    if missing:
        return False, f"explanations missing models: {sorted(missing)}"
    for field in MODEL_FIELDS:
        items = explanations[field]
        if not isinstance(items, list) or not (1 <= len(items) <= 5):
            return False, f"explanations.{field} expected 1-5 entries, got {len(items) if isinstance(items, list) else items!r}"
        try:
            magnitudes = [abs(float(i["contribution"])) for i in items]
        except (KeyError, TypeError, ValueError):
            return False, f"explanations.{field} entries lack numeric 'contribution': {items!r}"
        if magnitudes != sorted(magnitudes, reverse=True):
            return False, (
                f"explanations.{field} not sorted by |contribution| desc: {magnitudes}"
            )
    return True, "explanations: 1-5 entries per model, sorted by |contribution| desc"


def _check_reply_nonempty(check, scenario, status, body, runner):
    reply = body.get("reply") if isinstance(body, dict) else None
    ok = isinstance(reply, str) and bool(reply.strip())
    return ok, f"reply expected non-empty string, got {reply!r}"


def _check_reply_mentions_tier(check, scenario, status, body, runner):
    language = _request_language(scenario)
    labels = TIER_LABELS.get(language)
    if labels is None:
        return False, f"no tier labels for language '{language}'"
    tier = _context_tier(scenario)
    label = labels[tier]
    reply = body.get("reply") if isinstance(body, dict) else None
    if not isinstance(reply, str):
        return False, f"reply is not a string: {reply!r}"
    if language in _CJK_LANGUAGES:
        ok = label in reply
    else:
        ok = re.search(rf"\b{re.escape(label)}\b", reply, re.IGNORECASE) is not None
    return ok, (
        f"reply expected to mention overall tier '{tier}' as '{label}' "
        f"(language={language}); reply={reply!r}"
    )


def _check_sources_urls_allowed(check, scenario, status, body, runner):
    """回复里的每个链接都必须出现在本轮 sources 中（可选地约束 sources 条数）。

    这是「不许编造引用」在端到端层面的门禁：sources 是工具真正返回过的东西，
    回复只能引用它们。min_sources / max_sources 用来分别固化「问了地点就该有
    来源」和「没问地点就不该有来源」两种场景。
    """
    if not isinstance(body, dict):
        return False, f"response body is not an object (HTTP {status})"
    sources = body.get("sources")
    if not isinstance(sources, list):
        return False, f"response has no sources list (HTTP {status}); body keys={sorted(body)}"

    allowed = [str(s.get("url", "")) for s in sources if isinstance(s, dict)]
    minimum, maximum = check.get("min_sources"), check.get("max_sources")
    if minimum is not None and len(sources) < minimum:
        return False, f"expected at least {minimum} source(s), got {len(sources)}"
    if maximum is not None and len(sources) > maximum:
        return False, f"expected at most {maximum} source(s), got {len(sources)}: {allowed}"

    reply = body.get("reply")
    cited = [u.rstrip(URL_TRAILING) for u in URL_RE.findall(str(reply or ""))]
    unlisted = [u for u in cited if u not in allowed]
    if unlisted:
        return False, f"reply cites URL(s) absent from sources: {unlisted}; sources={allowed}"
    return True, (
        f"{len(cited)} cited URL(s) all present in {len(sources)} source(s): {allowed}"
    )


def _check_advice_source(check, scenario, status, body, runner):
    """advice_source 必须如实说明这段文字是模型写的还是模板兜底的。"""
    expect = check.get("expect")
    if expect not in ADVICE_SOURCES:
        return False, f"advice_source check needs expect in {list(ADVICE_SOURCES)}, got {expect!r}"
    got = body.get("advice_source") if isinstance(body, dict) else None
    return got == expect, f"advice_source expected {expect!r}, got {got!r}"


def _check_scores_match_scenario(check, scenario, status, body, runner):
    ref_id = check.get("ref")
    ref = runner.by_id.get(ref_id)
    if ref is None:
        return False, f"referenced scenario '{ref_id}' not found in scenarios file"
    ref_status, ref_body = runner.call(ref)
    if ref_status != 200 or not isinstance(ref_body, dict):
        return False, f"referenced scenario '{ref_id}' did not return HTTP 200 ({ref_status})"
    if status != 200 or not isinstance(body, dict):
        return False, f"this scenario did not return HTTP 200 ({status})"

    diffs = []
    for field in MODEL_FIELDS:
        z_here = body.get(field, {}).get("z")
        z_ref = ref_body.get(field, {}).get("z")
        if (
            not isinstance(z_here, (int, float))
            or not isinstance(z_ref, (int, float))
            or abs(z_here - z_ref) > Z_TOLERANCE
        ):
            diffs.append(f"{field}.z: {z_here} != {z_ref}")
    if diffs:
        return False, f"z values differ from scenario '{ref_id}': " + "; ".join(diffs)
    return True, f"all three z values match scenario '{ref_id}'"


CHECKS = {
    "status": _check_status,
    "level": _check_level,
    "score_between": _check_score_between,
    "warning_signs": _check_warning_signs,
    "exposure_level": _check_exposure_level,
    "field_equals": _check_field_equals,
    "advice_order": _check_advice_order,
    "medical_urgency": _check_medical_urgency,
    "no_probability_language": _check_no_probability_language,
    "explanations_present": _check_explanations_present,
    "reply_nonempty": _check_reply_nonempty,
    "reply_mentions_tier": _check_reply_mentions_tier,
    "sources_urls_allowed": _check_sources_urls_allowed,
    "advice_source": _check_advice_source,
    "scores_match_scenario": _check_scores_match_scenario,
}


# ---------- 运行器 ----------


class ScenarioRunner:
    """按需执行场景请求并缓存响应（scores_match_scenario 需要引用别的场景）。"""

    def __init__(self, client, scenarios: list[dict]) -> None:
        self.client = client
        self.by_id = {s["id"]: s for s in scenarios}
        self._cache: dict[str, tuple[int, object]] = {}

    def call(self, scenario: dict) -> tuple[int, object]:
        sid = scenario["id"]
        if sid not in self._cache:
            path = ENDPOINTS[scenario.get("endpoint", "assess")]
            resp = self.client.post(path, json=scenario["request"])
            try:
                parsed = resp.json()
            except ValueError:
                parsed = None
            self._cache[sid] = (resp.status_code, parsed)
        return self._cache[sid]

    def evaluate(self, scenario: dict) -> dict:
        status, body = self.call(scenario)
        checks = []
        for check in scenario.get("checks", []):
            ctype = check.get("type")
            fn = CHECKS.get(ctype)
            if fn is None:
                ok, detail = False, f"unknown check type '{ctype}'"
            else:
                try:
                    ok, detail = fn(check, scenario, status, body, self)
                except Exception as exc:  # 检查自身崩溃也算失败，而不是把整轮跑挂掉
                    ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
            checks.append({"type": ctype, "ok": bool(ok), "detail": detail})
        return {
            "id": scenario["id"],
            "endpoint": scenario.get("endpoint", "assess"),
            "description": scenario.get("description", ""),
            "status": status,
            "passed": all(c["ok"] for c in checks),
            "checks": checks,
        }


def _clear_failures_dir(failures_dir: Path) -> None:
    """建目录并清掉上一轮的失败转储——目录里只保留本轮的失败案例。"""
    failures_dir.mkdir(parents=True, exist_ok=True)
    for stale in failures_dir.glob("*.json"):
        stale.unlink()


def _dump_failure(failures_dir: Path, scenario: dict, result: dict, body) -> Path:
    """失败案例落盘：请求 + 响应 + 失败的检查，足以离线复盘。"""
    record = {
        "id": scenario["id"],
        "description": scenario.get("description", ""),
        "endpoint": scenario.get("endpoint", "assess"),
        "request": scenario["request"],
        "status": result["status"],
        "response": body,
        "failed_checks": [c for c in result["checks"] if not c["ok"]],
    }
    path = failures_dir / f"{scenario['id']}.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def run_scenarios(
    scenarios: list[dict],
    only: list[str] | None = None,
    failures_dir: Path = DEFAULT_FAILURES_DIR,
) -> dict:
    """执行（可过滤的）场景集合，返回汇总结果字典。"""
    selected = scenarios
    if only:
        by_id = {s["id"]: s for s in scenarios}
        unknown = [sid for sid in only if sid not in by_id]
        if unknown:
            raise ValueError(f"--only 指定了不存在的场景 id：{unknown}")
        selected = [by_id[sid] for sid in only]

    _clear_failures_dir(failures_dir)

    results = []
    dumps: list[str] = []
    with build_client() as client:
        runner = ScenarioRunner(client, scenarios)
        for scenario in selected:
            result = runner.evaluate(scenario)
            results.append(result)
            if not result["passed"]:
                _, body = runner.call(scenario)
                dumps.append(str(_dump_failure(failures_dir, scenario, result, body)))

    passed = sum(1 for r in results if r["passed"])
    return {
        "total": len(results),
        "passed": passed,
        "failed": [r["id"] for r in results if not r["passed"]],
        "failure_dumps": dumps,
        "results": results,
    }


# ---------- 输出 ----------


def _symbols() -> tuple[str, str]:
    """Windows 控制台编码兜底：emoji 编码不了就退回 [PASS]/[FAIL]。"""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✅❌".encode(encoding)
        return "✅", "❌"
    except (UnicodeEncodeError, LookupError):
        return "[PASS]", "[FAIL]"


def _safe_print(text: str) -> None:
    """打印含中文/emoji 的行时不因控制台编码而崩溃。"""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding))


def print_report(summary: dict, failures_dir: Path) -> None:
    pass_mark, fail_mark = _symbols()
    for result in summary["results"]:
        mark = pass_mark if result["passed"] else fail_mark
        _safe_print(f"{mark} {result['id']}")
        if not result["passed"]:
            for check in result["checks"]:
                if not check["ok"]:
                    _safe_print(f"    - {check['type']}: {check['detail']}")
    _safe_print(f"passed {summary['passed']}/{summary['total']}")
    if summary["failed"]:
        _safe_print(f"failure library: {len(summary['failed'])} dump(s) in {failures_dir}")
        for path in summary["failure_dumps"]:
            _safe_print(f"    {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="登革热风险服务场景化评测运行器")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON 结果")
    parser.add_argument(
        "--only",
        default="",
        help="只运行指定场景，逗号分隔的 id 列表，如 --only a,b",
    )
    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIOS),
        help="场景文件路径（默认 service/eval/scenarios.json）",
    )
    parser.add_argument(
        "--failures-dir",
        default=str(DEFAULT_FAILURES_DIR),
        help="失败案例转储目录（默认 service/eval/failures）",
    )
    args = parser.parse_args(argv)

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.is_file():
        print(f"场景文件不存在：{scenarios_path}", file=sys.stderr)
        return 2
    try:
        scenarios = load_scenarios(scenarios_path)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"场景文件非法：{exc}", file=sys.stderr)
        return 2

    only = [sid.strip() for sid in args.only.split(",") if sid.strip()]
    try:
        summary = run_scenarios(
            scenarios, only=only or None, failures_dir=Path(args.failures_dir)
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        _safe_print(json.dumps({"scenarios": str(scenarios_path), **summary}, ensure_ascii=True, indent=2))
    else:
        print_report(summary, Path(args.failures_dir))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
