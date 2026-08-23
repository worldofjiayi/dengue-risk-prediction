"""The three-layer data contract: questionnaire input FormInput -> ML features
MLFeatures -> assessment result AssessmentResult.

The feature definitions align strictly with the training script of the dengue risk
model (the FEATS list in 02_fit_models.py); their order and naming must not change,
or inference results become meaningless.

Rationale for the three-state answer (yes/no/unknown) encoding: the SINAN training
data uses 1=yes, 2=no, 9=unknown, and the feature engineering step only counts "1"
as 1 -- so as far as the model is concerned, "no" and "don't know" are both 0.

Warning: the epidemiological exposure questions (EXPOSURE_CODES) are **not model
features**. The SINAN notification data does not contain those three variables, and
forcing them into the 26-dimensional vector would break alignment with the training
script. They travel down the "rule-based" channel only, producing a separate
ExposureContext; see that class for details.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------- Feature definitions (identical to the training script; order is fixed) ----------

# 14 symptoms
SYMPTOM_CODES: tuple[str, ...] = (
    "FEBRE",       # fever
    "MIALGIA",     # myalgia
    "CEFALEIA",    # headache
    "EXANTEMA",    # rash
    "VOMITO",      # vomiting
    "NAUSEA",      # nausea
    "DOR_COSTAS",  # back pain
    "CONJUNTVIT",  # conjunctivitis
    "ARTRITE",     # arthritis
    "ARTRALGIA",   # arthralgia (joint pain)
    "PETEQUIA_N",  # petechiae (pinpoint skin haemorrhages)
    "LEUCOPENIA",  # leukopenia (the strongest predictor of severe disease)
    "LACO",        # positive tourniquet test
    "DOR_RETRO",   # retro-orbital pain
)

# 7 comorbidities
COMORB_CODES: tuple[str, ...] = (
    "DIABETES",    # diabetes
    "HEMATOLOG",   # haematological disease
    "HEPATOPAT",   # liver disease
    "RENAL",       # kidney disease
    "HIPERTENSA",  # hypertension
    "ACIDO_PEPT",  # peptic ulcer disease
    "AUTO_IMUNE",  # autoimmune disease
)

# 5 non-binary features (keep their own names in the explanation output; no _x stripping)
NON_BINARY_FEATS: tuple[str, ...] = ("age", "sex_f", "day_ill", "wk_sin", "wk_cos")

# The complete ordering of the 26 features (= FEATS at training time)
FEATS: tuple[str, ...] = (
    tuple(f"{c}_x" for c in SYMPTOM_CODES)
    + tuple(f"{c}_x" for c in COMORB_CODES)
    + NON_BINARY_FEATS
)

# ---------- Epidemiological exposure (rule-based channel, **not** model features) ----------
#
# These three questions simply do not exist in the SINAN notification data, so they
# cannot enter the logistic regression model: a variable never seen during training has
# no coefficient, and forcing one in would knock the 26-dimensional vector out of
# alignment with the training script.
# But "there is a confirmed case around me" and "I have been to an outbreak area" are
# among the most important clues in a dengue history, and it would be a shame to throw
# them away. The compromise is to put them behind a **separate rule** (see
# pipeline.evaluate_exposure), presented alongside the model score without interfering
# with it -- the same design used for the WHO warning signs.
EXPOSURE_CODES: tuple[str, ...] = (
    "FEVER_CLUSTER",    # an unusual recent rise in fever cases among people nearby
    "CONFIRMED_CASE",   # a confirmed dengue case close by (household / workplace / community)
    "OUTBREAK_TRAVEL",  # recently visited or lived in a dengue outbreak area
)

# Exposure factors that make the level high (any one of them being yes is enough)
HIGH_EXPOSURE_CODES: tuple[str, ...] = ("CONFIRMED_CASE", "OUTBREAK_TRAVEL")
# Exposure factors that make the level medium (only when high was not reached)
MEDIUM_EXPOSURE_CODES: tuple[str, ...] = ("FEVER_CLUSTER",)

# ---------- Enumerated types ----------

SymptomAnswer = Literal["yes", "no", "unknown"]
Sex = Literal["F", "M"]
RiskLevel = Literal["low", "medium", "high"]
Language = Literal["zh-CN", "zh-TW", "en", "es", "pt"]
ModelKey = Literal["A", "B", "B2"]
# Where the advice text came from: real model output (having passed output validation),
# or the built-in template fallback
AdviceSource = Literal["llm", "template"]
# Which layer a cited source came from: the WHO Disease Outbreak News API, or the
# model's web search
SourceOrigin = Literal["who", "search"]
SourceAuthority = Literal["official", "other"]
# Status of web search for this round:
#   ok       -- search ran and brought back sources
#   degraded -- search was attempted but failed / found nothing / the output did not pass
#               validation (every other layer still returns as usual)
#   disabled -- SEARCH_ENABLED=false, no search was ever going to be attempted
SearchStatus = Literal["ok", "degraded", "disabled"]
# Regional endemicity level (matching the values in app/data/dengue_endemicity.json)
Endemicity = Literal["high", "moderate", "low", "none", "unknown"]

# ---------- Fixed copy in five languages ----------

DISCLAIMERS: dict[str, str] = {
    "zh-CN": "本结果仅供参考，不构成医疗诊断。如有不适请及时就医。",
    "zh-TW": "本結果僅供參考，不構成醫療診斷。如有不適請及時就醫。",
    "en": (
        "This result is for reference only and does not constitute a medical diagnosis. "
        "Please seek medical care if you feel unwell."
    ),
    "es": (
        "Este resultado es solo orientativo y no constituye un diagnóstico médico. "
        "Si se siente mal, busque atención médica."
    ),
    "pt": (
        "Este resultado é apenas para referência e não constitui um diagnóstico médico. "
        "Se não se sentir bem, procure atendimento médico."
    ),
}

# A note on what the model is: a relative score rather than a probability (no intercept
# plus downsampled training).
#
# The thresholds are still uncalibrated for any local population, but that caveat was
# dropped from this user-facing string after a design review: on the result page it sat
# next to the score and read as hedging rather than as information a member of the public
# could act on. It remains documented where a reader can act on it -- both READMEs, the
# technical report, and app/ml_model.py's module docstring.
MODEL_NOTES: dict[str, str] = {
    "zh-CN": (
        "评分为相对风险参考值，非感染概率。模型基于巴西 SINAN 2023–2025 年"
        "登革热通报数据训练。"
    ),
    "zh-TW": (
        "評分為相對風險參考值，非感染機率。模型基於巴西 SINAN 2023–2025 年"
        "登革熱通報資料訓練。"
    ),
    "en": (
        "Scores are relative risk indicators, not infection probabilities. The model was "
        "trained on Brazilian SINAN dengue surveillance data (2023–2025)."
    ),
    "es": (
        "Las puntuaciones son indicadores de riesgo relativo, no probabilidades de infección. "
        "El modelo se entrenó con datos de vigilancia de dengue de SINAN Brasil (2023–2025)."
    ),
    "pt": (
        "As pontuações são indicadores de risco relativo, não probabilidades de infecção. "
        "O modelo foi treinado com dados de vigilância de dengue do SINAN Brasil (2023–2025)."
    ),
}

# Default copy (Simplified Chinese)
DISCLAIMER = DISCLAIMERS["zh-CN"]

# Message returned to the user when the upstream model is unavailable (HTTP 502)
UPSTREAM_ERRORS: dict[str, str] = {
    "zh-CN": "上游模型服务暂时不可用，请稍后重试。",
    "zh-TW": "上游模型服務暫時無法使用，請稍後重試。",
    "en": "The upstream model service is temporarily unavailable. Please try again shortly.",
    "es": "El servicio del modelo no está disponible temporalmente. Inténtelo de nuevo más tarde.",
    "pt": "O serviço do modelo está temporariamente indisponível. Tente novamente em instantes.",
}

# Internal server error message (HTTP 500)
SERVER_ERRORS: dict[str, str] = {
    "zh-CN": "服务器内部错误，请稍后重试。",
    "zh-TW": "伺服器內部錯誤，請稍後重試。",
    "en": "Internal server error. Please try again later.",
    "es": "Error interno del servidor. Inténtelo de nuevo más tarde.",
    "pt": "Erro interno do servidor. Tente novamente mais tarde.",
}

# The WHO dengue warning signs (Guidelines for Diagnosis, Treatment, Prevention and
# Control, 2009) that this questionnaire is able to cover. This is a **rule-based check
# independent of the model**: models B/B2 are dominated by leukopenia, so a patient who
# has not had blood work done can score low even when warning signs are already present.
# The rule-based alert must therefore be shown alongside the score, so that users are
# not falsely reassured.
WARNING_SIGN_CODES: tuple[str, ...] = (
    "VOMITO",      # persistent vomiting
    "PETEQUIA_N",  # mucosal / skin bleeding
)


class FormInput(BaseModel):
    """POST /api/assess request body: the dengue risk self-assessment questionnaire."""

    age: int = Field(..., ge=0, le=110, description="age (years)")
    sex: Sex = Field(..., description="sex at birth, F female / M male")
    day_ill: int = Field(..., ge=0, le=14, description="days since symptom onset")
    # validate_default=True: the fill-in validator must run even when the whole field is
    # absent, otherwise form.symptoms would be an empty dict and downstream lookups by
    # code would raise KeyError.
    symptoms: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        validate_default=True,
        description="the 14 symptoms; missing keys are treated as unknown",
    )
    comorbidities: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        validate_default=True,
        description="the 7 comorbidities; missing keys are treated as unknown",
    )
    exposure: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        validate_default=True,
        description=(
            "the 3 epidemiological exposure questions; missing keys are treated as "
            "unknown. **Takes no part in model scoring**, and is only used for the "
            "rule-based exposure_context."
        ),
    )
    language: Language = Field(default="zh-CN", description="output language (BCP 47)")
    notes: str = Field(default="", max_length=500, description="optional free-text notes")

    @field_validator("symptoms")
    @classmethod
    def _fill_symptoms(cls, value: dict[str, str]) -> dict[str, str]:
        return _fill_answers(value, SYMPTOM_CODES, "symptoms")

    @field_validator("comorbidities")
    @classmethod
    def _fill_comorbidities(cls, value: dict[str, str]) -> dict[str, str]:
        return _fill_answers(value, COMORB_CODES, "comorbidities")

    @field_validator("exposure")
    @classmethod
    def _fill_exposure(cls, value: dict[str, str]) -> dict[str, str]:
        return _fill_answers(value, EXPOSURE_CODES, "exposure")


def _fill_answers(value: dict, codes: tuple[str, ...], field: str) -> dict:
    """Fill missing keys in as unknown; raise on any key outside the contract."""
    _require_known_keys(value, codes, field)
    return {code: value.get(code, "unknown") for code in codes}


def _require_known_keys(value: dict, codes: tuple[str, ...], field: str) -> dict:
    """Only validate that keys are within the contract; **do not fill missing keys in**.

    The semantics of /api/plan are "key present = answered, key absent = not asked yet",
    so absence is itself the signal and must never be filled in as unknown the way
    FormInput does.
    """
    unknown_keys = set(value) - set(codes)
    if unknown_keys:
        raise ValueError(f"{field} contains unknown keys: {sorted(unknown_keys)}")
    return value


class MLFeatures(BaseModel):
    """The 26 model input features. Binary items are 0/1; the rest are continuous."""

    # 14 symptoms
    FEBRE_x: int = Field(..., ge=0, le=1)
    MIALGIA_x: int = Field(..., ge=0, le=1)
    CEFALEIA_x: int = Field(..., ge=0, le=1)
    EXANTEMA_x: int = Field(..., ge=0, le=1)
    VOMITO_x: int = Field(..., ge=0, le=1)
    NAUSEA_x: int = Field(..., ge=0, le=1)
    DOR_COSTAS_x: int = Field(..., ge=0, le=1)
    CONJUNTVIT_x: int = Field(..., ge=0, le=1)
    ARTRITE_x: int = Field(..., ge=0, le=1)
    ARTRALGIA_x: int = Field(..., ge=0, le=1)
    PETEQUIA_N_x: int = Field(..., ge=0, le=1)
    LEUCOPENIA_x: int = Field(..., ge=0, le=1)
    LACO_x: int = Field(..., ge=0, le=1)
    DOR_RETRO_x: int = Field(..., ge=0, le=1)
    # 7 comorbidities
    DIABETES_x: int = Field(..., ge=0, le=1)
    HEMATOLOG_x: int = Field(..., ge=0, le=1)
    HEPATOPAT_x: int = Field(..., ge=0, le=1)
    RENAL_x: int = Field(..., ge=0, le=1)
    HIPERTENSA_x: int = Field(..., ge=0, le=1)
    ACIDO_PEPT_x: int = Field(..., ge=0, le=1)
    AUTO_IMUNE_x: int = Field(..., ge=0, le=1)
    # continuous / other
    age: float = Field(..., ge=0.0, le=110.0)
    sex_f: float = Field(..., ge=0.0, le=1.0)
    day_ill: float = Field(..., ge=0.0, le=14.0)
    wk_sin: float = Field(..., ge=-1.0, le=1.0)
    wk_cos: float = Field(..., ge=-1.0, le=1.0)

    def as_vector(self) -> list[float]:
        """Expand into a feature vector in FEATS order (for use by an external model)."""
        data = self.model_dump()
        return [float(data[name]) for name in FEATS]


class ModelScore(BaseModel):
    """Output of a single model: a relative risk score."""

    score: float = Field(..., ge=0.0, le=100.0, description="0-100 relative score")
    level: RiskLevel
    z: float = Field(..., description="linear predictor (no intercept)")


class ExposureContext(BaseModel):
    """Epidemiological exposure context: **the result of a rule, not of any model**.

    level:
        high   -- CONFIRMED_CASE or OUTBREAK_TRAVEL is yes
        medium -- FEVER_CLUSTER is yes and high was not reached
        low    -- everything else (including all-"don't know")
    factors: the exposure codes answered yes, for the front end to look up and display
    localised labels.

    Why this is not folded into the model score: these three variables do not exist in
    the SINAN training data, so there is no coefficient to use, and any weighting would
    be a number pulled out of thin air. Presenting them separately is the honest option.
    """

    level: RiskLevel
    factors: list[str] = Field(default_factory=list)


class FeatureContribution(BaseModel):
    """One feature's contribution to a model's linear predictor z (coef × feature value)."""

    feature: str = Field(..., description="feature name from FEATS, e.g. FEBRE_x")
    code: str = Field(
        ...,
        description=(
            "label key for the front end to look up: symptoms/comorbidities drop the _x "
            "suffix, the 5 non-binary features keep their own names"
        ),
    )
    contribution: float = Field(..., description="coef × feature value, to 4 decimal places")
    direction: Literal["up", "down"] = Field(..., description="pushes the score up or down")


class Advice(BaseModel):
    """Three kinds of advice, each a list of strings in the target language
    (FormInput.language).

    Field order is display order in the front end: **seeking care comes first**, then
    home monitoring, then day-to-day protection. Pydantic serialises in field
    declaration order, so changing this directly changes the key order of the response
    JSON.
    """

    medical: list[str]
    monitoring: list[str]
    protection: list[str]


class AssessmentResult(BaseModel):
    """POST /api/assess response body."""

    dengue: ModelScore      # model A: dengue or not
    worsening: ModelScore   # model B: worsening or not (warning + severe vs ordinary)
    severe: ModelScore      # model B2: severe or not
    epi_week: int = Field(..., ge=1, le=52, description="epidemiological week of the assessment date")
    warning_signs: list[str] = Field(
        default_factory=list,
        description=(
            "WHO dengue warning-sign codes reported by the user (rule-based, "
            "independent of the model score)"
        ),
    )
    exposure_context: ExposureContext = Field(
        ...,
        description="epidemiological exposure context (rule-based, independent of the model score)",
    )
    summary: str
    advice: Advice
    explanations: dict[str, list[FeatureContribution]] = Field(
        default_factory=dict,
        description="top 5 contributions to each model's z, keyed by dengue / worsening / severe",
    )
    disclaimer: str = DISCLAIMER
    model_note: str = MODEL_NOTES["zh-CN"]
    advice_source: AdviceSource = Field(
        default="template",
        description=(
            "who wrote this advice: llm = generated by the real model and passed output "
            "validation; template = the built-in template (demo mode, or the fallback "
            "after the model failed or failed validation twice). The scores and the "
            "rule-based checks are genuinely computed in both cases; only the natural "
            "language part differs."
        ),
    )


# ---------- Follow-up chat (POST /api/chat) ----------

# History message cap: keep only the most recent N, silently truncating the rest
# (friendlier than a 422)
CHAT_HISTORY_MAX = 6
CHAT_QUESTION_MAX = 500


class ChatScore(BaseModel):
    """A single model's score echoed back by the front end (a trimmed-down
    AssessmentResult.ModelScore)."""

    score: float = Field(..., ge=0.0, le=100.0)
    level: RiskLevel


class ChatContext(BaseModel):
    """A snapshot of the user's own assessment result. The server is stateless, so the
    front end echoes all of it back."""

    dengue: ChatScore | None = None
    worsening: ChatScore | None = None
    severe: ChatScore | None = None
    warning_signs: list[str] = Field(default_factory=list)
    exposure_level: RiskLevel = "low"
    symptoms: dict[str, SymptomAnswer] = Field(default_factory=dict)
    comorbidities: dict[str, SymptomAnswer] = Field(default_factory=dict)
    age: int | None = Field(default=None, ge=0, le=110)
    sex: Sex | None = None
    day_ill: int | None = Field(default=None, ge=0, le=14)

    @field_validator("symptoms")
    @classmethod
    def _keep_known_symptoms(cls, value: dict[str, str]) -> dict[str, str]:
        return _drop_unknown_keys(value, SYMPTOM_CODES)

    @field_validator("comorbidities")
    @classmethod
    def _keep_known_comorbidities(cls, value: dict[str, str]) -> dict[str, str]:
        return _drop_unknown_keys(value, COMORB_CODES)

    @field_validator("warning_signs")
    @classmethod
    def _keep_known_warning_signs(cls, value: list[str]) -> list[str]:
        return [c for c in value if c in WARNING_SIGN_CODES]


class ChatMessage(BaseModel):
    """One turn of message history."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    """POST /api/chat request body."""

    language: Language = "zh-CN"
    question: str = Field(..., min_length=1, max_length=CHAT_QUESTION_MAX)
    context: ChatContext = Field(default_factory=ChatContext)
    history: list[ChatMessage] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question must not be blank")
        return text

    @field_validator("history")
    @classmethod
    def _truncate_history(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        # Truncate rather than raise: a user should not be blocked by a 422 just for
        # having had a long conversation
        return value[-CHAT_HISTORY_MAX:]


class Source(BaseModel):
    """A single citable source.

    origin says **which layer** produced this link:
        who    -- a notice app.intel fetched from the WHO Disease Outbreak News API
                  (stable, free, always available)
        search -- a page the model's web search actually returned this round (metered,
                  and may come back with nothing)
    The two layers differ in trustworthiness and freshness and the front end has to be
    able to label them separately, so the label travels with the link rather than being
    guessed from a URL prefix.

    authority is a second distinction **within a single layer**: search results place a
    national health ministry's notice and a news aggregator side by side (measured, in
    the Singapore case: nea.gov.sg listed alongside magzter.com). They are worlds apart
    in how checkable they are, so we judge by domain whether a source is a government or
    international health body and let the front end mark it. The judgement looks only at
    the domain, never at the content -- that keeps it a checkable fact rather than an
    assessment of quality.

    date is allowed to be empty: WHO notices always have a publication date, search
    results frequently do not.
    """

    title: str = Field(..., description="page title, taken verbatim from the API/search result")
    date: str | None = Field(default=None, description="publication date; null when unavailable")
    url: str = Field(..., description="source page address")
    origin: SourceOrigin = Field(default="who", description="which layer this source came from")
    authority: SourceAuthority = Field(
        default="other",
        description="whether the domain belongs to a government or international health body",
    )


class ChatResponse(BaseModel):
    """POST /api/chat response body.

    sources holds the sources **actually retrieved this round**: the union of the WHO
    tool results and the web search results. It is simultaneously the citation list
    shown to the user and the allow-list the verifier uses to decide whether "a link in
    the reply was made up" -- if any link appears in the reply that is not in this list,
    the round is judged a failure and falls back to the template copy.
    When nothing was looked up (or nothing was found) it is an empty list, and the reply
    should not contain any links either.
    """

    reply: str
    sources: list[Source] = Field(default_factory=list)
    search_count: int = Field(
        default=0,
        ge=0,
        description=(
            "how many web searches really ran this round. Always 0 when no location was "
            "recognised or SEARCH_ENABLED=false -- search is metered, and whether money "
            "was spent should not be knowable only from the server log."
        ),
    )


# ---------- Destination lookup (POST /api/destination) ----------

DESTINATION_LOCATION_MAX = 120


class WhoNotice(BaseModel):
    """One WHO Disease Outbreak News notice (same shape as
    intel.lookup_dengue_context)."""

    title: str
    date: str
    url: str


class DestinationAdvice(BaseModel):
    """The three kinds of pre-travel advice.

    The field order is **deliberately different from Advice**: there is no patient here
    and no score, the user is asking "what is it like there right now" before leaving.
    The first thing to say is how not to get bitten; only then "when to see a doctor"
    and what to watch for during the trip. Pydantic serialises in declaration order, so
    this order is the front end's display order.
    """

    protection: list[str]
    medical: list[str]
    monitoring: list[str]


class DestinationRequest(BaseModel):
    """POST /api/destination request body."""

    location: str = Field(
        ...,
        min_length=1,
        max_length=DESTINATION_LOCATION_MAX,
        description="country / region / city name, in any language",
    )
    language: Language = Field(default="zh-CN", description="output language (BCP 47)")

    @field_validator("location")
    @classmethod
    def _location_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("location must not be blank")
        return text


class DestinationResponse(BaseModel):
    """POST /api/destination response body.

    **There is no score here**, not one. A location never takes part in scoring and
    never changes the exposure band; it is pre-travel reference material, a different
    thing entirely from the three model outputs. Insisting on returning a "destination
    risk score" here would mean inventing a number off the back of a coarse-grained
    country table.

    Three layers, in descending order of trustworthiness:
      1. endemicity / season_note / who_notices -- local table + WHO API, stable and free;
      2. recent_findings -- "last three months" points from the model's web search,
         metered, and may be absent;
      3. advice -- fixed copy matched to the band, always available.
    When layer 2 does not come back we degrade to 1+3, and never pad it out with
    "common knowledge".
    """

    location: str = Field(..., description="canonical English name; the raw input if unrecognised")
    matched: bool = Field(..., description="whether the place name was found in the built-in region table")
    endemicity: Endemicity
    season_note: str | None = Field(
        default=None, description="seasonal/regional note; null when there was no match"
    )
    who_notices: list[WhoNotice] = Field(default_factory=list)
    recent_findings: list[str] = Field(
        default_factory=list,
        description=(
            "factual points from roughly the last three months; an empty list when the "
            "search failed or did not pass validation"
        ),
    )
    sources: list[Source] = Field(
        default_factory=list,
        description=(
            "merged, de-duplicated list of WHO notices (origin=who) and search results "
            "(origin=search)"
        ),
    )
    advice: DestinationAdvice
    search_status: SearchStatus
    disclaimer: str = DISCLAIMER
    model_note: str = MODEL_NOTES["zh-CN"]


# Maximum number of search sources shown to the user per request.
# Measured: two rounds of search brought back 2 × 10 = 20 results, more than half of
# them news aggregators and irrelevant pages. Twenty links is not a citation list, it
# is noise.
MAX_SEARCH_SOURCES = 8


def select_search_sources(
    sources: list[dict] | None, reply: str, limit: int = MAX_SEARCH_SOURCES
) -> list[dict]:
    """Pick, out of the pile of results the search returned, the ones to display (which
    is to say, the ones that go into the allow-list).

    **Not one link the reply actually cited may be dropped**: sources is both the
    citation list and the verifier's allow-list, so truncating away the very entry the
    model cited would get the reply judged as "fabricated link" by our own verifier, and
    a good answer would be rolled back to the template copy. So we collect every cited
    source first, then top up to limit in the original order.
    """
    items = [s for s in (sources or []) if isinstance(s, dict) and s.get("url")]
    text = reply or ""
    cited = [s for s in items if str(s["url"]) in text]
    chosen = list(cited)
    seen = {str(s["url"]) for s in cited}
    for item in items:
        if len(chosen) >= max(limit, len(cited)):
            break
        url = str(item["url"])
        if url not in seen:
            seen.add(url)
            chosen.append(item)
    return chosen


# Government domain labels: these appear immediately before a country top-level domain
# (nea.gov.sg / doh.gov.ph / moph.go.th / gob.mx / gouv.fr / govt.nz). A bare "go" is
# dangerous on its own (go.com is not a government), so it only counts when followed by
# a two-letter country code.
_GOV_LABELS = frozenset({"gov", "gob", "go", "gouv", "govt"})

# Bodies that are genuinely international health/public institutions without a
# government domain, listed separately. Better to under-classify than to
# over-classify: marking something official tells the user "this one is more
# checkable", and getting that wrong costs more than missing one.
_OFFICIAL_HOSTS = frozenset(
    {
        "who.int",
        "paho.org",
        "europa.eu",       # ECDC lives at ecdc.europa.eu
        "un.org",
        "unicef.org",
    }
)


def classify_authority(url: str) -> SourceAuthority:
    """Judge from the domain whether this source is a government / international health body.

    Only the domain is examined, never the content -- that way the conclusion is a
    checkable fact rather than an assessment of reporting quality. A health ministry's
    outbreak notice and a news aggregator's reprint are things the reader is entitled to
    tell apart at a glance.

    Rules (any one match makes it official):
      - the top-level domain is gov or int          -- cdc.gov / who.int
      - the second-to-last label is a government label and the TLD is a two-letter
        country code                                -- nea.gov.sg / moph.go.th / gob.mx
      - the domain or one of its parents is in _OFFICIAL_HOSTS  -- ecdc.europa.eu
    """
    raw = (url or "").strip().lower()
    if "//" in raw:
        raw = raw.split("//", 1)[1]
    host = raw.split("/", 1)[0].split("?", 1)[0].split("@")[-1].split(":", 1)[0]
    host = host.rstrip(".")
    if not host:
        return "other"

    labels = [p for p in host.split(".") if p]
    if len(labels) < 2:
        return "other"

    if labels[-1] in {"gov", "int"}:
        return "official"
    if len(labels[-1]) == 2 and labels[-2] in _GOV_LABELS:
        return "official"
    for i in range(len(labels) - 1):
        if ".".join(labels[i:]) in _OFFICIAL_HOSTS:
            return "official"
    return "other"


def merge_sources(
    who_notices: list[dict] | None, search_sources: list[dict] | None
) -> list[Source]:
    """Combine the two layers of sources into one origin-tagged list: WHO notices first,
    search results after.

    This lives in schemas rather than in some pipeline module because both /api/chat and
    /api/destination need it, and "what a source looks like and which layer it came
    from" is part of the data contract in the first place.

    De-duplicated by url, order preserved; WHO comes first because it is both more
    stable and easier to check.
    date is None when unavailable (search results often have no page_age) -- never
    substitute today's date for it.

    Within the search segment, official sources are then moved to the front: whether the
    model cites them is its own business, but "which of these came from a health
    authority" should be the first thing to catch the reader's eye. **Reorder only,
    never drop** -- sources is also the verifier's allow-list, and removing any entry
    could get a correct reply judged as having fabricated a link.
    """
    merged: list[Source] = []
    seen: set[str] = set()
    for notice in who_notices or []:
        url = str(notice.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        merged.append(
            Source(
                title=str(notice.get("title") or url),
                date=str(notice.get("date") or "") or None,
                url=url,
                origin="who",
                authority="official",  # a WHO notice is an official source by definition
            )
        )
    found: list[Source] = []
    for item in search_sources or []:
        url = str(item.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        date = item.get("date")
        found.append(
            Source(
                title=str(item.get("title") or url),
                date=str(date) if date else None,
                url=url,
                origin="search",
                authority=classify_authority(url),
            )
        )
    # Stable sort: official first, everything else keeps the order search returned it in
    found.sort(key=lambda s: 0 if s.authority == "official" else 1)
    merged.extend(found)
    return merged


def _drop_unknown_keys(value: dict, codes: tuple[str, ...]) -> dict:
    """Keep only the keys that are within the contract.

    Unlike FormInput's strict validation: the /api/chat context is a snapshot echoed
    back by the front end, and one unfamiliar extra key should not stop the user asking
    a question. Silently dropping it is enough.
    """
    return {k: v for k, v in value.items() if k in codes}


# ---------- Adaptive questioning plan (POST /api/plan) ----------


class PlanRequest(BaseModel):
    """POST /api/plan request body: a **partially answered** questionnaire.

    The key difference from FormInput: **the presence or absence of a key is itself
    information**.
      - key present = the question has been asked (yes / no / unknown are all definite
        answers);
      - key absent = the question has not been asked yet, and its eventual value is
        undetermined.

    FormInput therefore cannot be reused -- its validators fill missing keys in as
    unknown, which erases exactly the distinction between "answered don't know" and "not
    asked yet", and that distinction is the planner's entire basis for working.
    age / sex / day_ill are required in the first step of the questionnaire, so planning
    starts from them already being known.
    """

    age: int = Field(..., ge=0, le=110, description="age (years)")
    sex: Sex = Field(..., description="sex at birth, F female / M male")
    day_ill: int = Field(..., ge=0, le=14, description="days since symptom onset")
    symptoms: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        description="answered symptoms: key present = asked. Missing key = not asked; not filled in.",
    )
    comorbidities: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        description="answered comorbidities; same semantics as symptoms.",
    )
    language: Language = Field(
        default="zh-CN", description="output language (used to localise error messages)"
    )

    @field_validator("symptoms")
    @classmethod
    def _known_symptom_keys(cls, value: dict[str, str]) -> dict[str, str]:
        return _require_known_keys(value, SYMPTOM_CODES, "symptoms")

    @field_validator("comorbidities")
    @classmethod
    def _known_comorb_keys(cls, value: dict[str, str]) -> dict[str, str]:
        return _require_known_keys(value, COMORB_CODES, "comorbidities")


class ModelBounds(BaseModel):
    """Hard score bounds for a single model under partial answers (same normalisation,
    same banding)."""

    score_now: float = Field(
        ..., ge=0.0, le=100.0,
        description=(
            "current score counting unasked questions as 0 -- the final score if the "
            "user stopped answering right now"
        ),
    )
    score_min: float = Field(
        ..., ge=0.0, le=100.0,
        description="lower bound: no answer to the remaining questions can go below this",
    )
    score_max: float = Field(
        ..., ge=0.0, le=100.0,
        description="upper bound: no answer to the remaining questions can go above this",
    )
    level_now: RiskLevel
    decided: bool = Field(
        ..., description="whether [score_min, score_max] already falls within a single risk band"
    )


class PlanBounds(BaseModel):
    """Score bounds for the three models, keyed the same way as AssessmentResult."""

    dengue: ModelBounds
    worsening: ModelBounds
    severe: ModelBounds


class NextQuestion(BaseModel):
    """A question suggested as the next one to ask."""

    kind: Literal["symptom", "comorbidity"]
    code: str = Field(..., description="question code from SYMPTOM_CODES / COMORB_CODES")
    why_model: Literal["dengue", "worsening", "severe"] = Field(
        ...,
        description="which not-yet-banded model's estimate this question mainly narrows",
    )


class PlanResponse(BaseModel):
    """POST /api/plan response body."""

    bounds: PlanBounds
    can_stop: bool = Field(
        ...,
        description="all three models decided: no remaining answer can change any band",
    )
    next: list[NextQuestion] = Field(
        default_factory=list,
        description=(
            "at most 5, in descending order of information value; always empty once "
            "every model is banded"
        ),
    )
    answered: int = Field(..., ge=0, description="number of symptoms + comorbidities answered")
    remaining: int = Field(..., ge=0, description="number of symptoms + comorbidities not yet asked")
