"""Output verifier: **pure rule** checks on LLM-generated text; no model call, no network.

Its place: the last gate after the model has generated and before anything reaches the
user. It does not judge whether the advice is any good, only whether it crossed a line this
service does not allow crossing -- the lines that can be written down as rules:

  dosage          a specific drug dose (this service never prescribes; a dose must come
                  from a clinician)
  probability     presenting a percentage as "your probability of infection" (the model has
                  no intercept, only relative scores)
  urgency_missing high risk / warning signs present, yet no mention of seeking care (the
                  most dangerous kind of failure)
  language_mismatch  answering in the wrong language (the user chose Spanish, the reply
                  came back in Chinese)
  structure       shape out of bounds (empty items, too many items, one item too long)
  fabricated_url  citing a link no tool returned this turn (a fabricated citation)
  empty           empty reply

Every violation carries a message written **for the model**: the pipeline splices it back
into the prompt and asks for one rewrite, falling back to the template copy if that still
comes out wrong. The message therefore uses the second person and spells out what is wrong
and how to fix it.

⚠️ The seek-care lexicon in this module is deliberately **not** imported from
scripts/eval_run.py, nor the other way round. The two lexicons are duplicated on purpose:
if either side quietly changes its wording, the other one goes red.
"""

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Keys of the three advice sections (matching the fields of schemas.Advice)
ADVICE_SECTIONS: tuple[str, ...] = ("medical", "monitoring", "protection")

# Character cap for one advice item / allowed item-count range per section
MAX_ITEM_CHARS = 400
MIN_ITEMS = 1
MAX_ITEMS = 5

# CJK detection: used by the language consistency check
_CJK_RE = re.compile(r"[㐀-䶿一-鿿]")

# Minimum CJK character share required of the Chinese languages; maximum share allowed
# for the Latin-script ones
CJK_MIN_RATIO = 0.3
CJK_MAX_RATIO = 0.05

# ---------- Rule 1: dosage ----------
# "number + dose unit" or "dosing frequency". Note that only **a number directly adjacent to
# a unit** counts: a medication warning carrying no number, such as "avoid taking aspirin or
# ibuprofen", must be let through -- that is a safety note, not a prescription.
# (?![0-9A-Za-z]) keeps g from matching words like gums / gently.
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

# ---------- Rule 2: infection probability ----------
_PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%|百分之\s*\d+")
_PROBABILITY_RE = re.compile(
    r"概率|機率|\bprobability\b|\bchance\b|\bprobabilidad\b|\bprobabilidade\b",
    re.IGNORECASE,
)
# Second-person reference: only a sentence carrying a percentage AND probability wording AND
# "you" counts as pinning a probability on the user. "90% of dengue cases are mild" is an
# epidemiological fact and must not be caught by mistake.
_SECOND_PERSON_RE = re.compile(
    r"您|你|\byou\b|\byour\b|\busted\b|\bsu\b|\bvocê\b|\bvoce\b|\bseu\b",
    re.IGNORECASE,
)
# Sentence splitting: sentence-final punctuation in zh/en/es/pt + newlines + semicolons
_SENTENCE_SPLIT_RE = re.compile(r"[。！？；!?;\n]+|(?<=[a-zA-Z0-9\)\]])\.(?=\s|$)")

# ---------- Rule 3: seek-care lexicon (**deliberately duplicated in eval_run.py**) ----------
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

# ---------- Rule 4: language consistency ----------
CJK_LANGUAGES: tuple[str, ...] = ("zh-CN", "zh-TW")
FUNCTION_WORDS: dict[str, tuple[str, ...]] = {
    "es": ("que", "para", "con", "los", "las"),
    "pt": ("que", "para", "com", "dos", "uma", "não"),
    "en": ("the", "and", "your", "with"),
}
MIN_FUNCTION_WORDS = 2

# ---------- Rule 6: URL ----------
_URL_RE = re.compile(r"https?://[^\s<>\"'）)】\]\[（(，。；、]+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?'\")]}>，。；！？、）】"


@dataclass(frozen=True)
class Violation:
    """One violation. code is what the program branches on; message is fed back to the
    model verbatim to ask for a rewrite.
    """

    code: str
    message: str

    def __str__(self) -> str:  # convenient for logs and prompt splicing
        return f"[{self.code}] {self.message}"


# ---------- Shared helpers ----------


def _texts_of_advice(advice: Any) -> dict[str, list[str]]:
    """Normalise an Advice model or an equivalent dict into {section: [item, ...]}."""
    sections: dict[str, list[str]] = {}
    for name in ADVICE_SECTIONS:
        if isinstance(advice, Mapping):
            items = advice.get(name)
        else:
            items = getattr(advice, name, None)
        if isinstance(items, str):  # catch a bare string too; the structure rule reports it
            items = [items]
        sections[name] = [str(i) for i in items] if isinstance(items, Sequence) and not isinstance(items, str) else []
    return sections


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _cjk_ratio(text: str) -> float:
    """Share of CJK ideographs among **non-whitespace characters**. Empty text counts as 0."""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(1 for c in dense if _CJK_RE.match(c)) / len(dense)


def extract_urls(text: str) -> list[str]:
    """Extract the http(s) links in a reply, stripping trailing punctuation."""
    found = []
    for raw in _URL_RE.findall(text or ""):
        url = raw.rstrip(_URL_TRAILING)
        if url:
            found.append(url)
    return found


# ---------- Individual rules ----------


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
    if keywords is None:  # unknown language: nothing to judge against, so no false alarm
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


# ---------- Public interface ----------


def verify_advice(
    advice: Any,
    summary: str,
    language: str,
    overall_tier: str,
    warning_signs: Sequence[str] | None = None,
) -> list[Violation]:
    """Verify a generated advice object. Returns a list of violations, empty when all pass.

    advice may be a schemas.Advice instance or a dict of the same shape -- the verifier has
    to run both before and after Pydantic validation.
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
    """Verify one follow-up reply.

    allowed_urls are the links **this turn's tool calls actually returned**. An empty list
    means there is no citable source this turn, so any link appearing in the reply counts
    as fabricated -- this is precisely the invariant that stops the model from "recalling"
    a WHO page of its own.
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
    """Splice the violation list into the correction request fed back to the model.

    as_json=True is for advice generation (the output is a JSON object), False for
    follow-up replies (prose).
    """
    listed = "\n".join(f"- {v}" for v in violations)
    shape = "the same JSON object" if as_json else "your answer as plain prose"
    return (
        "Your previous answer violated the following output rules:\n"
        f"{listed}\n"
        f"Regenerate {shape}, fixing only these issues. Keep everything else unchanged."
    )
