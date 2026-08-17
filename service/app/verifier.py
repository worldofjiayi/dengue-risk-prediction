"""输出校验器：对 LLM 生成的文本做**纯规则**检查，不调用任何模型、不发网络请求。

定位：模型生成之后、返回给用户之前的最后一道闸门。它不判断「建议好不好」，
只判断「有没有越过这个服务不允许越过的线」——那些线是可以用规则写死的：

  dosage          具体药物剂量（本服务从不开处方，剂量必须由医生给）
  probability     把百分比说成「你感染的概率」（模型无截距，只有相对评分）
  urgency_missing 高风险 / 有警示征象却没说要就医（最危险的一种失败）
  language_mismatch  答非所语（用户选了西语却回中文）
  structure       结构越界（条目为空、条目过多、单条过长）
  fabricated_url  引用了本轮工具没返回过的链接（编造引用）
  empty           空回复

每条违规都带一个**给模型看的** message：流水线会把它拼回提示词里要求重写一次，
写不对就退回模板文案。因此 message 用第二人称、说清「错在哪、该怎么改」。

⚠️ 本模块的就医词表刻意**不**从 scripts/eval_run.py 导入，反之亦然。
两份词表是故意重复的：任何一方悄悄改了措辞，另一方就会报红。
"""

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# 三类建议的键（与 schemas.Advice 的字段一致）
ADVICE_SECTIONS: tuple[str, ...] = ("medical", "monitoring", "protection")

# 单条建议的字符上限 / 每类建议的条数区间
MAX_ITEM_CHARS = 400
MIN_ITEMS = 1
MAX_ITEMS = 5

# CJK 判定：用于语言一致性检查
_CJK_RE = re.compile(r"[㐀-䶿一-鿿]")

# 中文类语言要求的最低 CJK 字符占比；拉丁语言允许的最高占比
CJK_MIN_RATIO = 0.3
CJK_MAX_RATIO = 0.05

# ---------- 规则 1：剂量 ----------
# 「数字 + 剂量单位」或「服药频次」。注意只认**数字紧邻单位**的写法：
# 「避免服用阿司匹林或布洛芬」这种不带数字的用药提醒必须放行——那是安全提示，
# 不是处方。(?![0-9A-Za-z]) 防止 g 命中 gums / gently 这类单词。
_DOSAGE_UNIT_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*"
    r"(?:mg|ml|mcg|µg|ug|g(?![0-9A-Za-z])|grams?|gramas?|gramos?"
    r"|comprimidos?|tabletas?|tablets?|片|粒|毫克|毫升|微克|克(?!服))",
    re.IGNORECASE,
)
_DOSAGE_FREQ_RE = re.compile(
    r"(?:每\s*\d+\s*(?:小时|小時|个小时|個小時)"
    r"|a\s+cada\s+\d+\s*horas?"
    r"|cada\s+\d+\s*horas?"
    r"|every\s+\d+\s*hours?)",
    re.IGNORECASE,
)

# ---------- 规则 2：感染概率 ----------
_PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%|百分之\s*\d+")
_PROBABILITY_RE = re.compile(
    r"概率|機率|\bprobability\b|\bchance\b|\bprobabilidad\b|\bprobabilidade\b",
    re.IGNORECASE,
)
# 第二人称指代：句子里同时出现「百分比 + 概率措辞 + 你」才算把概率安到用户头上。
# 「90% 的登革热病例是轻症」是流行病学事实，不能误伤。
_SECOND_PERSON_RE = re.compile(
    r"您|你|\byou\b|\byour\b|\busted\b|\bsu\b|\bvocê\b|\bvoce\b|\bseu\b",
    re.IGNORECASE,
)
# 断句：中英西葡的句末标点 + 换行 + 分号
_SENTENCE_SPLIT_RE = re.compile(r"[。！？；!?;\n]+|(?<=[a-zA-Z0-9\)\]])\.(?=\s|$)")

# ---------- 规则 3：就医紧迫性词表（**故意与 eval_run.py 重复**） ----------
URGENCY_LEXICON: dict[str, tuple[str, ...]] = {
    "zh-CN": ("尽快就医", "立即就医", "及时就医", "尽早就医", "就医", "就诊", "急诊", "前往医院"),
    "zh-TW": ("儘快就醫", "盡快就醫", "立即就醫", "及時就醫", "就醫", "就診", "急診", "前往醫院"),
    "en": (
        "seek medical", "seek care", "medical care", "medical attention",
        "see a clinician", "see a doctor", "medical review",
        "emergency department", "urgent care",
    ),
    "es": (
        "busque atención médica", "atención médica", "acuda", "consulte",
        "consulta médica", "servicio de urgencias", "urgencias",
    ),
    "pt": (
        "procure atendimento", "atendimento médico", "procure um profissional",
        "avaliação médica", "pronto-socorro", "unidade de saúde",
    ),
}

# ---------- 规则 4：语言一致性 ----------
CJK_LANGUAGES: tuple[str, ...] = ("zh-CN", "zh-TW")
FUNCTION_WORDS: dict[str, tuple[str, ...]] = {
    "es": ("que", "para", "con", "los", "las"),
    "pt": ("que", "para", "com", "dos", "uma", "não"),
    "en": ("the", "and", "your", "with"),
}
MIN_FUNCTION_WORDS = 2

# ---------- 规则 6：URL ----------
_URL_RE = re.compile(r"https?://[^\s<>\"'）)】\]\[（(，。；、]+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?'\")]}>，。；！？、）】"


@dataclass(frozen=True)
class Violation:
    """一条违规。code 供程序分支，message 会被原样喂回模型要求重写。"""

    code: str
    message: str

    def __str__(self) -> str:  # 便于日志与提示词拼接
        return f"[{self.code}] {self.message}"


# ---------- 通用工具 ----------


def _texts_of_advice(advice: Any) -> dict[str, list[str]]:
    """把 Advice 模型或等价 dict 归一成 {section: [item, ...]}。"""
    sections: dict[str, list[str]] = {}
    for name in ADVICE_SECTIONS:
        if isinstance(advice, Mapping):
            items = advice.get(name)
        else:
            items = getattr(advice, name, None)
        if isinstance(items, str):  # 单条字符串也接住，交给 structure 规则报错
            items = [items]
        sections[name] = [str(i) for i in items] if isinstance(items, Sequence) and not isinstance(items, str) else []
    return sections


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _cjk_ratio(text: str) -> float:
    """CJK 表意字符占**非空白字符**的比例。空文本记 0。"""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(1 for c in dense if _CJK_RE.match(c)) / len(dense)


def extract_urls(text: str) -> list[str]:
    """抽取回复中的 http(s) 链接，去掉尾随标点。"""
    found = []
    for raw in _URL_RE.findall(text or ""):
        url = raw.rstrip(_URL_TRAILING)
        if url:
            found.append(url)
    return found


# ---------- 单条规则 ----------


def _check_dosage(texts: Sequence[str], where: str) -> list[Violation]:
    for text in texts:
        hit = _DOSAGE_UNIT_RE.search(text) or _DOSAGE_FREQ_RE.search(text)
        if hit:
            return [
                Violation(
                    "dosage",
                    f"You wrote a specific drug dose or dosing frequency ({hit.group(0)!r}) in the {where}. "
                    "This service never prescribes. Remove every number attached to a dose unit "
                    "(mg, ml, g, tablets, 片, 粒) and every dosing interval, and say that any "
                    "medication must be directed by a clinician.",
                )
            ]
    return []


def _check_probability(texts: Sequence[str], where: str) -> list[Violation]:
    for text in texts:
        for sentence in _sentences(text):
            if (
                _PERCENT_RE.search(sentence)
                and _PROBABILITY_RE.search(sentence)
                and _SECOND_PERSON_RE.search(sentence)
            ):
                return [
                    Violation(
                        "probability",
                        f"In the {where} you presented a percentage as this person's own probability "
                        f"of infection: {sentence.strip()!r}. The three scores are relative risk "
                        "indicators from a model fitted without an intercept — they are not "
                        "probabilities. Delete the figure and describe the level in words "
                        "(relatively low / moderate / relatively high).",
                    )
                ]
    return []


def _check_urgency(
    medical: Sequence[str], language: str, overall_tier: str, warning_signs: Sequence[str]
) -> list[Violation]:
    if overall_tier != "high" and not warning_signs:
        return []
    keywords = URGENCY_LEXICON.get(language)
    if keywords is None:  # 未知语言：无从判断，不误报
        return []
    lowered = [str(item).lower() for item in medical]
    if any(k.lower() in text for text in lowered for k in keywords):
        return []
    reason = (
        "the overall risk tier is high"
        if overall_tier == "high"
        else f"the person reported WHO warning signs ({', '.join(warning_signs)})"
    )
    return [
        Violation(
            "urgency_missing",
            f"None of your advice.medical items tells the person to seek medical care, but {reason}. "
            f"Rewrite advice.medical so its first item plainly says to see a clinician "
            f"(in {language}, e.g. one of: {', '.join(keywords[:3])}).",
        )
    ]


def _check_language(text: str, language: str) -> list[Violation]:
    ratio = _cjk_ratio(text)
    if language in CJK_LANGUAGES:
        if ratio < CJK_MIN_RATIO:
            return [
                Violation(
                    "language_mismatch",
                    f"The answer must be written in {language}, but only {ratio:.0%} of it is Chinese "
                    "characters. Rewrite the whole answer in that language.",
                )
            ]
        return []

    words = FUNCTION_WORDS.get(language)
    if words is None:
        return []
    if ratio > CJK_MAX_RATIO:
        return [
            Violation(
                "language_mismatch",
                f"The answer must be written in {language}, but {ratio:.0%} of it is Chinese characters. "
                "Rewrite the whole answer in that language.",
            )
        ]
    lowered = text.lower()
    present = {w for w in words if re.search(rf"(?<![0-9a-zà-ÿ]){re.escape(w)}(?![0-9a-zà-ÿ])", lowered)}
    if len(present) < MIN_FUNCTION_WORDS:
        return [
            Violation(
                "language_mismatch",
                f"The answer does not read as {language}: it contains fewer than {MIN_FUNCTION_WORDS} "
                f"of the expected common words ({', '.join(words)}). Rewrite the whole answer in that language.",
            )
        ]
    return []


def _check_structure(sections: Mapping[str, list[str]]) -> list[Violation]:
    problems: list[str] = []
    for name in ADVICE_SECTIONS:
        items = [i for i in sections.get(name, []) if i and i.strip()]
        if not (MIN_ITEMS <= len(items) <= MAX_ITEMS):
            problems.append(
                f"advice.{name} has {len(items)} non-blank items (need {MIN_ITEMS}-{MAX_ITEMS})"
            )
        for index, item in enumerate(items):
            if len(item) > MAX_ITEM_CHARS:
                problems.append(
                    f"advice.{name}[{index}] is {len(item)} characters (max {MAX_ITEM_CHARS})"
                )
    if not problems:
        return []
    return [
        Violation(
            "structure",
            "The advice object has the wrong shape: "
            + "; ".join(problems)
            + f". Every one of {', '.join(ADVICE_SECTIONS)} needs {MIN_ITEMS}-{MAX_ITEMS} "
            f"non-empty items of at most {MAX_ITEM_CHARS} characters.",
        )
    ]


# ---------- 对外接口 ----------


def verify_advice(
    advice: Any,
    summary: str,
    language: str,
    overall_tier: str,
    warning_signs: Sequence[str] | None = None,
) -> list[Violation]:
    """校验一份生成的建议。返回违规列表，全部通过时为空列表。

    advice 可以是 schemas.Advice 实例，也可以是同形状的 dict——校验器要能在
    Pydantic 校验之前/之后都跑得动。
    """
    warning_signs = list(warning_signs or [])
    sections = _texts_of_advice(advice)
    all_items = [item for items in sections.values() for item in items]
    scanned = [*all_items, summary or ""]

    violations: list[Violation] = []
    violations += _check_dosage(scanned, "advice")
    violations += _check_probability(scanned, "advice")
    violations += _check_urgency(sections["medical"], language, overall_tier, warning_signs)
    violations += _check_language("\n".join(scanned), language)
    violations += _check_structure(sections)
    return violations


def verify_chat_reply(
    reply: str, language: str, allowed_urls: Sequence[str] | None = None
) -> list[Violation]:
    """校验一条追问回复。

    allowed_urls 是**本轮工具调用真正返回过的**链接。空列表意味着这一轮没有
    任何可引用来源，于是回复里出现任何链接都算编造——这正是阻止模型自己
    「想」出一个 WHO 页面的那条不变量。
    """
    text = reply or ""
    if not text.strip():
        return [
            Violation(
                "empty",
                "Your reply was empty. Answer the question in one or two short paragraphs.",
            )
        ]

    allowed = [u for u in (allowed_urls or []) if u]
    violations: list[Violation] = []
    violations += _check_dosage([text], "reply")
    violations += _check_probability([text], "reply")

    fabricated = [u for u in extract_urls(text) if not any(u.startswith(a) for a in allowed)]
    if fabricated:
        allowed_note = (
            "No source was returned by any tool this turn, so the reply must contain no links at all."
            if not allowed
            else "The only links you may cite are: " + ", ".join(allowed)
        )
        violations.append(
            Violation(
                "fabricated_url",
                f"Your reply cites {', '.join(fabricated)}, which no tool returned this turn. "
                + allowed_note
                + " Remove every other link; never reconstruct a URL from memory.",
            )
        )
    return violations


def format_violations(violations: Sequence[Violation], *, as_json: bool = True) -> str:
    """把违规列表拼成回喂给模型的整改要求。

    as_json=True 用于建议生成（输出是 JSON 对象），False 用于追问回复（散文）。
    """
    listed = "\n".join(f"- {v}" for v in violations)
    shape = "the same JSON object" if as_json else "your answer as plain prose"
    return (
        "Your previous answer violated the following output rules:\n"
        f"{listed}\n"
        f"Regenerate {shape}, fixing only these issues. Keep everything else unchanged."
    )
