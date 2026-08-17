"""Output verifier (app/verifier.py) tests: every rule is tested in **both directions**.

Testing only that "a violation is caught" is not enough -- a verifier that always
returned a violation would pass that too, and it would send every real generation back
to the template. So every rule also gets a sample that **must be allowed through**,
once in each of the five languages: the edges of the wording are exactly where these
rules are most likely to misfire.

The ballast is the last one: built-in templates × 5 languages × 3 bands × with/without
warning signs, and the violation count must be 0. The templates are what gets served to
the user when verification fails; if they are not clean themselves, the whole fallback
chain is pointless.
"""

import pytest

from app.deepseek_client import fallback_advice
from app.verifier import (
    MAX_ITEM_CHARS,
    Violation,
    extract_urls,
    format_violations,
    verify_advice,
    verify_chat_reply,
)

LANGUAGES = ("zh-CN", "zh-TW", "en", "es", "pt")
TIERS = ("low", "medium", "high")


def base_advice(language: str, tier: str = "medium") -> tuple[dict, str]:
    """A guaranteed-clean advice block + summary (the very template used as fallback)."""
    raw = fallback_advice(language, tier)
    return {k: list(v) for k, v in raw["advice"].items()}, raw["summary"]


def codes(violations) -> set[str]:
    return {v.code for v in violations}


def check(language: str, *, medical_extra: str = "", summary_extra: str = "",
          tier: str = "medium", warning_signs=()) -> set[str]:
    """Inject one sentence under test into a clean template; return the violation codes.

    Changing one sentence and leaving the rest clean keeps the assertion on the rule
    being tested: any other code means the injected sentence tripped something else as
    well, and the test shows it immediately.
    """
    advice, summary = base_advice(language, tier)
    if medical_extra:
        advice["medical"] = [*advice["medical"][:1], medical_extra]
    if summary_extra:
        summary = summary + " " + summary_extra
    return codes(verify_advice(advice, summary, language, tier, list(warning_signs)))


# ---------- Rule 1: dosage ----------

# "Avoid aspirin/ibuprofen" is a safety note that **must pass**: no numbers, not a
# prescription.
NO_DOSE_PHRASES = {
    "zh-CN": "退热镇痛请遵医嘱，避免自行服用阿司匹林或布洛芬类药物。",
    "zh-TW": "退燒止痛請遵醫囑，避免自行服用阿斯匹靈或布洛芬類藥物。",
    "en": "Avoid aspirin or ibuprofen unless a clinician tells you otherwise.",
    "es": "Evite la aspirina o el ibuprofeno salvo indicación médica.",
    "pt": "Evite aspirina ou ibuprofeno, salvo orientação médica.",
}

WITH_DOSE_PHRASES = {
    "zh-CN": "请每 6 小时服用对乙酰氨基酚 500 毫克。",
    "zh-TW": "請每 6 小時服用乙醯胺酚 500 毫克。",
    "en": "Take 500 mg of paracetamol every 6 hours.",
    "es": "Tome 500 mg de paracetamol cada 6 horas.",
    "pt": "Tome 500 mg de paracetamol a cada 6 horas.",
}


@pytest.mark.parametrize("language", LANGUAGES)
def test_avoiding_a_drug_without_numbers_is_not_a_dosage(language):
    assert "dosage" not in check(language, medical_extra=NO_DOSE_PHRASES[language])


@pytest.mark.parametrize("language", LANGUAGES)
def test_numeric_dose_is_flagged(language):
    assert "dosage" in check(language, medical_extra=WITH_DOSE_PHRASES[language])


@pytest.mark.parametrize(
    "text",
    ["Take two 500 mg tablets", "服用 2 片退热药", "每8小时一次", "cada 8 horas", "a cada 8 horas"],
)
def test_dosage_variants_all_caught(text):
    assert "dosage" in codes(verify_chat_reply(text, "en", []))


@pytest.mark.parametrize(
    "text",
    [
        "The 24-48 hours after the fever drops is the critical window.",
        "Las 24-48 horas posteriores a la caída de la fiebre son críticas.",
        "退热后 24-48 小时是关键窗口，请每日测量 2 次体温。",
        "Rest for 3 days and drink plenty of fluids.",
    ],
)
def test_ordinary_numbers_are_not_dosages(text):
    assert "dosage" not in codes(verify_chat_reply(text, "en", []))


# ---------- Rule 2: infection probability ----------

# Pinning a percentage on **this person** -- must be blocked
PERSONAL_PROBABILITY = {
    "zh-CN": "您感染登革热的概率大约是 42%。",
    "zh-TW": "您感染登革熱的機率大約是 42%。",
    "en": "You have a 42% probability of infection.",
    "es": "Usted tiene una probabilidad del 42% de estar infectado.",
    "pt": "Você tem 42% de probabilidade de estar infectado.",
}

# A population-level statistical fact -- must be allowed
POPULATION_STATISTIC = {
    "zh-CN": "在所有登革热病例中，约 90% 属于轻症。",
    "zh-TW": "在所有登革熱病例中，約 90% 屬於輕症。",
    "en": "90% of dengue cases are mild.",
    "es": "El 90% de los casos de dengue son leves.",
    "pt": "90% dos casos de dengue são leves.",
}


@pytest.mark.parametrize("language", LANGUAGES)
def test_personal_infection_probability_is_flagged(language):
    assert "probability" in check(language, summary_extra=PERSONAL_PROBABILITY[language])


@pytest.mark.parametrize("language", LANGUAGES)
def test_population_statistic_is_allowed(language):
    assert "probability" not in check(language, summary_extra=POPULATION_STATISTIC[language])


def test_probability_needs_all_three_signals_in_one_sentence():
    """Percentage, probability wording and second person must share one sentence to count."""
    assert "probability" not in codes(verify_chat_reply("Your risk is 42%.", "en", []))
    assert "probability" not in codes(
        verify_chat_reply("The probability of severe dengue is low for you.", "en", [])
    )
    # Split across two sentences does not count either: a statistic plus talking to the
    # user is not "your probability is X%"
    assert "probability" not in codes(
        verify_chat_reply("90% of cases are mild. Your scores are relative.", "en", [])
    )
    assert "probability" in codes(
        verify_chat_reply("Your chance of infection is about 42%.", "en", [])
    )


# ---------- Rule 3: urgency of seeking care ----------

NO_URGENCY_MEDICAL = {
    "zh-CN": "多喝水、好好休息就可以了。",
    "zh-TW": "多喝水、好好休息就可以了。",
    "en": "Rest at home and drink plenty of fluids.",
    "es": "Descanse en casa y beba muchos líquidos.",
    "pt": "Descanse em casa e beba bastante líquido.",
}


@pytest.mark.parametrize("language", LANGUAGES)
def test_high_tier_without_seek_care_is_flagged(language):
    advice, summary = base_advice(language, "high")
    advice["medical"] = [NO_URGENCY_MEDICAL[language]]
    assert "urgency_missing" in codes(verify_advice(advice, summary, language, "high", []))


@pytest.mark.parametrize("language", LANGUAGES)
def test_high_tier_template_says_seek_care(language):
    advice, summary = base_advice(language, "high")
    assert "urgency_missing" not in codes(verify_advice(advice, summary, language, "high", []))


@pytest.mark.parametrize("language", LANGUAGES)
def test_warning_signs_demand_urgency_even_at_low_tier(language):
    """Warning signs are independent of the score: even a low score must say seek care."""
    advice, summary = base_advice(language, "low")
    advice["medical"] = [NO_URGENCY_MEDICAL[language]]
    got = codes(verify_advice(advice, summary, language, "low", ["VOMITO"]))
    assert "urgency_missing" in got
    # The same text, with no warning signs and a low band, must not be required to urge
    assert "urgency_missing" not in codes(
        verify_advice(advice, summary, language, "low", [])
    )


# ---------- Rule 4: language consistency ----------


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("tier", TIERS)
def test_every_template_passes_its_own_language(language, tier):
    advice, summary = base_advice(language, tier)
    assert "language_mismatch" not in codes(
        verify_advice(advice, summary, language, tier, [])
    )


@pytest.mark.parametrize("wrong", ["en", "es", "pt"])
def test_chinese_text_declared_as_latin_language_is_flagged(wrong):
    advice, summary = base_advice("zh-CN", "medium")
    assert "language_mismatch" in codes(verify_advice(advice, summary, wrong, "medium", []))


@pytest.mark.parametrize("declared", ["zh-CN", "zh-TW"])
def test_latin_text_declared_as_chinese_is_flagged(declared):
    advice, summary = base_advice("en", "medium")
    assert "language_mismatch" in codes(
        verify_advice(advice, summary, declared, "medium", [])
    )


def test_wrong_latin_language_needs_two_function_words():
    """English text passed off as Spanish: two of que/para/con/los/las are missing."""
    advice, summary = base_advice("en", "medium")
    assert "language_mismatch" in codes(verify_advice(advice, summary, "es", "medium", []))
    # The other way round: a real Spanish template must not be judged as not Spanish
    es_advice, es_summary = base_advice("es", "medium")
    assert "language_mismatch" not in codes(
        verify_advice(es_advice, es_summary, "es", "medium", [])
    )


# ---------- Rule 5: structure ----------


def test_empty_section_is_flagged():
    advice, summary = base_advice("en", "low")
    advice["monitoring"] = []
    assert "structure" in codes(verify_advice(advice, summary, "en", "low", []))


def test_blank_strings_do_not_count_as_items():
    advice, summary = base_advice("en", "low")
    advice["protection"] = ["   ", ""]
    assert "structure" in codes(verify_advice(advice, summary, "en", "low", []))


def test_too_many_items_is_flagged():
    advice, summary = base_advice("en", "low")
    advice["protection"] = [f"Point number {i} about mosquito nets and the water." for i in range(6)]
    assert "structure" in codes(verify_advice(advice, summary, "en", "low", []))


def test_overlong_item_is_flagged():
    advice, summary = base_advice("en", "low")
    advice["monitoring"] = ["a" * (MAX_ITEM_CHARS + 1)]
    got = codes(verify_advice(advice, summary, "en", "low", []))
    assert "structure" in got
    # Exactly at the limit passes
    advice["monitoring"] = ["a" * MAX_ITEM_CHARS]
    assert "structure" not in codes(verify_advice(advice, summary, "en", "low", []))


def test_advice_may_be_a_pydantic_model_or_a_dict():
    """The verifier must run both before and after Pydantic validation."""
    from app.schemas import Advice

    raw, summary = base_advice("en", "medium")
    assert verify_advice(raw, summary, "en", "medium", []) == []
    assert verify_advice(Advice.model_validate(raw), summary, "en", "medium", []) == []


# ---------- Ballast: the templates are always clean ----------


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("warning_signs", [[], ["VOMITO", "PETEQUIA_N"]])
def test_every_mock_template_has_zero_violations(language, tier, warning_signs):
    """5 languages × 3 bands × with/without warning signs: zero violations in a template.

    This test guards the credibility of the fallback chain -- these are the very texts
    served to the user when the model fails.
    """
    advice, summary = base_advice(language, tier)
    assert verify_advice(advice, summary, language, tier, warning_signs) == []


# ---------- Chat replies: invented links ----------

WHO_URL = "https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON518"
OTHER_URL = "https://example.org/dengue"


def test_empty_reply_is_flagged():
    for text in ("", "   ", "\n"):
        assert codes(verify_chat_reply(text, "en", [WHO_URL])) == {"empty"}


def test_any_url_is_fabricated_when_no_tool_returned_anything():
    """An empty allow-list = no source at all this round, so every link is invented."""
    got = verify_chat_reply(f"See {WHO_URL} for details.", "en", [])
    assert codes(got) == {"fabricated_url"}
    assert "no tool returned" in got[0].message.lower() or "No source was returned" in got[0].message


def test_returned_url_is_allowed():
    assert verify_chat_reply(f"See {WHO_URL} for details.", "en", [WHO_URL]) == []


def test_url_must_prefix_match_an_allowed_one():
    allowed = ["https://www.who.int/emergencies/disease-outbreak-news/item/"]
    assert verify_chat_reply(f"See {WHO_URL}.", "en", allowed) == []
    assert "fabricated_url" in codes(verify_chat_reply(f"See {OTHER_URL}.", "en", allowed))


def test_one_good_url_does_not_excuse_a_bad_one():
    got = verify_chat_reply(f"Sources: {WHO_URL} and {OTHER_URL}", "en", [WHO_URL])
    assert "fabricated_url" in codes(got)
    assert OTHER_URL in got[0].message


def test_reply_with_no_urls_needs_no_sources():
    assert verify_chat_reply("Keep using repellent and rest.", "en", []) == []


def test_extract_urls_strips_trailing_punctuation():
    assert extract_urls(f"see {WHO_URL}.") == [WHO_URL]
    assert extract_urls(f"（来源：{WHO_URL}）") == [WHO_URL]
    assert extract_urls("no links here") == []


def test_chat_reply_checks_dosage_and_probability_too():
    got = codes(
        verify_chat_reply("You have a 42% probability of dengue. Take 500 mg every 6 hours.", "en", [])
    )
    assert got == {"dosage", "probability"}


# ---------- The violation messages themselves ----------


def test_violation_messages_are_fed_back_to_the_model():
    violations = [Violation("dosage", "no numbers"), Violation("probability", "no percentages")]
    text = format_violations(violations, as_json=True)
    assert "[dosage] no numbers" in text
    assert "[probability] no percentages" in text
    assert "JSON" in text
    assert "JSON" not in format_violations(violations, as_json=False)
