"""输出校验器（app/verifier.py）测试：每条规则都要**两个方向**都测。

只测「违规能被抓到」是不够的——一个永远返回违规的校验器也能通过那种测试，
而它会把每一次真实生成都打回模板。因此每条规则都配一条**必须放行**的样本，
而且五种语言各来一遍：措辞的边界正是这些规则最容易误伤的地方。

压舱石是最后那条：内置模板 × 5 语言 × 3 档位 × 有/无警示征象，违规数必须为 0。
模板是校验失败时端给用户的东西，它自己不干净，整条兜底链路就没有意义。
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
    """一份必然干净的建议 + summary（就是线上兜底用的那份模板）。"""
    raw = fallback_advice(language, tier)
    return {k: list(v) for k, v in raw["advice"].items()}, raw["summary"]


def codes(violations) -> set[str]:
    return {v.code for v in violations}


def check(language: str, *, medical_extra: str = "", summary_extra: str = "",
          tier: str = "medium", warning_signs=()) -> set[str]:
    """在干净模板上注入一句待测文本，返回违规码集合。

    只改一句、其余保持干净，是为了让断言真正落在被测规则上：
    出现别的 code 就说明注入的句子还踩了别的线，测试会立刻显形。
    """
    advice, summary = base_advice(language, tier)
    if medical_extra:
        advice["medical"] = [*advice["medical"][:1], medical_extra]
    if summary_extra:
        summary = summary + " " + summary_extra
    return codes(verify_advice(advice, summary, language, tier, list(warning_signs)))


# ---------- 规则 1：剂量 ----------

# 「避免阿司匹林/布洛芬」是**必须放行**的安全提示：它不带数字，不是处方。
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


# ---------- 规则 2：感染概率 ----------

# 把百分比安到**这个人**头上 —— 必须拦
PERSONAL_PROBABILITY = {
    "zh-CN": "您感染登革热的概率大约是 42%。",
    "zh-TW": "您感染登革熱的機率大約是 42%。",
    "en": "You have a 42% probability of infection.",
    "es": "Usted tiene una probabilidad del 42% de estar infectado.",
    "pt": "Você tem 42% de probabilidade de estar infectado.",
}

# 群体层面的统计事实 —— 必须放行
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
    """百分比、概率措辞、第二人称——三者同句才算数。"""
    assert "probability" not in codes(verify_chat_reply("Your risk is 42%.", "en", []))
    assert "probability" not in codes(
        verify_chat_reply("The probability of severe dengue is low for you.", "en", [])
    )
    # 分属两句也不算：统计事实 + 对用户说话，不构成「你的概率是 X%」
    assert "probability" not in codes(
        verify_chat_reply("90% of cases are mild. Your scores are relative.", "en", [])
    )
    assert "probability" in codes(
        verify_chat_reply("Your chance of infection is about 42%.", "en", [])
    )


# ---------- 规则 3：就医紧迫性 ----------

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
    """警示征象是独立于评分的：分数低也必须说去看医生。"""
    advice, summary = base_advice(language, "low")
    advice["medical"] = [NO_URGENCY_MEDICAL[language]]
    got = codes(verify_advice(advice, summary, language, "low", ["VOMITO"]))
    assert "urgency_missing" in got
    # 同一份文本，没有警示征象、档位也不高时不该被要求紧迫
    assert "urgency_missing" not in codes(
        verify_advice(advice, summary, language, "low", [])
    )


# ---------- 规则 4：语言一致性 ----------


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
    """英文文本冒充西语：缺少 que/para/con/los/las 中的两个。"""
    advice, summary = base_advice("en", "medium")
    assert "language_mismatch" in codes(verify_advice(advice, summary, "es", "medium", []))
    # 反向：真西语模板不该被判成不是西语
    es_advice, es_summary = base_advice("es", "medium")
    assert "language_mismatch" not in codes(
        verify_advice(es_advice, es_summary, "es", "medium", [])
    )


# ---------- 规则 5：结构 ----------


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
    # 刚好到上限则放行
    advice["monitoring"] = ["a" * MAX_ITEM_CHARS]
    assert "structure" not in codes(verify_advice(advice, summary, "en", "low", []))


def test_advice_may_be_a_pydantic_model_or_a_dict():
    """校验器要能在 Pydantic 校验前后都跑得动。"""
    from app.schemas import Advice

    raw, summary = base_advice("en", "medium")
    assert verify_advice(raw, summary, "en", "medium", []) == []
    assert verify_advice(Advice.model_validate(raw), summary, "en", "medium", []) == []


# ---------- 压舱石：模板永远干净 ----------


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("warning_signs", [[], ["VOMITO", "PETEQUIA_N"]])
def test_every_mock_template_has_zero_violations(language, tier, warning_signs):
    """5 语言 × 3 档位 × 有无警示征象：内置模板必须一条违规都没有。

    这条测试守的是兜底链路的可信度——模型失败时端出去的就是这些文本。
    """
    advice, summary = base_advice(language, tier)
    assert verify_advice(advice, summary, language, tier, warning_signs) == []


# ---------- 追问回复：编造链接 ----------

WHO_URL = "https://www.who.int/emergencies/disease-outbreak-news/item/2024-DON518"
OTHER_URL = "https://example.org/dengue"


def test_empty_reply_is_flagged():
    for text in ("", "   ", "\n"):
        assert codes(verify_chat_reply(text, "en", [WHO_URL])) == {"empty"}


def test_any_url_is_fabricated_when_no_tool_returned_anything():
    """空白名单 = 这轮没有任何来源，于是任何链接都是编的。"""
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


# ---------- 违规消息本身 ----------


def test_violation_messages_are_fed_back_to_the_model():
    violations = [Violation("dosage", "no numbers"), Violation("probability", "no percentages")]
    text = format_violations(violations, as_json=True)
    assert "[dosage] no numbers" in text
    assert "[probability] no percentages" in text
    assert "JSON" in text
    assert "JSON" not in format_violations(violations, as_json=False)
