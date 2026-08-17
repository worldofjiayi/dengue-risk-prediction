"""Dengue risk model inference.

== Where the model comes from ==
Three logistic regression models trained on 9.4499 million dengue notification
records from Brazil's SINAN notifiable-disease reporting system, 2023–2025.
The coefficients live in app/model/dengue_models.json:

  A  -- dengue or not (confirmed vs inconclusive)        AUC 0.686
  B  -- worsening or not (warning + severe vs ordinary)  AUC 0.722
  B2 -- severe or not (severe vs other dengue)           AUC 0.810 (the strongest)

== Three limitations you must know about ==

1. **No intercept.** The training script only exported coef_, not intercept_, so z
   is a linear predictor without a constant term. The 0-100 score derived from it
   can only be used as a **relative risk ranking**.

2. **Training used downsampling + class_weight="balanced".** Even if there were an
   intercept, it would correspond to the artificial post-resampling prevalence, not
   the true population prevalence. So the output of this module is **not an infection
   probability**, and no user-facing copy may present it as a percentage probability.

3. **The thresholds are uncalibrated.** The low/medium/high cut points are engineering
   defaults; before deployment to a real population they need to be re-evaluated on a
   test set that preserves the original prevalence (the wording of the "Known
   limitations" section of the project README).

== How the 0-100 score is derived ==

sigmoid(z) will not do: without an intercept z is persistently positive, and nearly
everyone with symptoms would approach 100.
Nor can the "theoretical minimum" serve as the lower bound: z_min corresponds to the
perverse state of "having every symptom with a negative coefficient", so a genuinely
asymptomatic person lands in the middle of the range instead (measured: a healthy
young person came out as medium).

We therefore use **reference-person anchoring**:

    score = 100 × (z − z_ref) / (z_ceil − z_ref)      result clipped to [0, 100]

    z_ref  = same season, no symptoms and no comorbidities, 30-year-old male, day 0 of illness
    z_ceil = same season, every risk-raising feature pushed to its upper bound
             (age 110, day_ill 14, binary features 1)

The score therefore means: **how high you sit on this model relative to an
asymptomatic person at this moment in time**.
The seasonal term is identical in z, z_ref and z_ceil, so it cancels out of the ratio
naturally -- and that is deliberate: wk_sin/wk_cos describe a population-level seasonal
baseline, not an individual risk difference.
(The seasonal term still enters the computation of z and is written to the evaluation
log, so that a future local calibration can use it.)

This is still only a **relative risk index**; it has nothing to do with probability.

== Feature encoding ==
The SINAN training data uses 1=yes, 2=no, 9=unknown, and the feature engineering step
(df[c] == "1") means that both "no" and "unknown" encode as 0. This module's
yes->1 / no->0 / unknown->0 matches that.

The three epidemiological exposure questions in the questionnaire (EXPOSURE_CODES)
**never reach this module**: the SINAN data does not contain those variables, so the
model has no coefficients for them. encode_features only reads
symptoms / comorbidities / age / sex / day_ill; the 26-dimensional vector is
independent of the exposure answers.

The seasonal terms wk_sin / wk_cos are computed by the server from the ISO week of the
assessment date; the user does not supply them.
Warning: the model was trained on southern-hemisphere (Brazilian) data, so the seasonal
term points the wrong way for northern-hemisphere users. Seasonality is itself weak near
the equator, which limits the impact, but this is a known transfer limitation.
"""

import json
import logging
import math
from datetime import date
from pathlib import Path

from app.schemas import (
    COMORB_CODES,
    FEATS,
    NON_BINARY_FEATS,
    SYMPTOM_CODES,
    FeatureContribution,
    FormInput,
    MLFeatures,
    ModelScore,
)

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).resolve().parent / "model" / "dengue_models.json"

# Risk band thresholds (engineering defaults, not calibrated against a population)
_LOW_MAX = 35.0
_MEDIUM_MAX = 65.0

# Model key -> result field name
MODEL_KEYS: tuple[str, ...] = ("A", "B", "B2")
RESULT_FIELDS: dict[str, str] = {"A": "dengue", "B": "worsening", "B2": "severe"}

# Maximum number of contribution items returned per model
EXPLAIN_TOP_N = 5


# Age of the reference person (anchors the zero point; does not follow the user's age)
_REF_AGE = 30.0
# Feature upper bounds (matching the validation ranges on FormInput)
_AGE_MAX = 110.0
_DAY_ILL_MAX = 14.0
# Seasonal feature names (kept as-is in both the reference person and the ceiling,
# so that they cancel out during normalisation)
_SEASON_FEATS = ("wk_sin", "wk_cos")


def _season_part(coef: dict[str, float], wk_sin: float, wk_cos: float) -> float:
    return coef.get("wk_sin", 0.0) * wk_sin + coef.get("wk_cos", 0.0) * wk_cos


def _reference_z(coef: dict[str, float], wk_sin: float, wk_cos: float) -> float:
    """Reference person: same season, no symptoms or comorbidities, 30-year-old male,
    day 0 of illness."""
    return coef.get("age", 0.0) * _REF_AGE + _season_part(coef, wk_sin, wk_cos)


def _ceiling_z(coef: dict[str, float], wk_sin: float, wk_cos: float) -> float:
    """Ceiling: same season, every risk-raising feature pushed to its upper bound."""
    z = _season_part(coef, wk_sin, wk_cos)
    for name in FEATS:
        if name in _SEASON_FEATS:
            continue
        c = coef.get(name, 0.0)
        if name == "age":
            z += max(0.0, c * _AGE_MAX)
        elif name == "day_ill":
            z += max(0.0, c * _DAY_ILL_MAX)
        else:  # binary features (including sex_f): take whichever of 0 or 1 contributes more
            z += max(0.0, c)
    return z


def _load_models() -> dict[str, dict]:
    """Load the model coefficient file; missing columns are treated as 0 at scoring
    time and logged as a warning."""
    with open(_MODEL_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    models: dict[str, dict] = {}
    for key in MODEL_KEYS:
        if key not in raw:
            raise ValueError(f"模型文件缺少模型 {key}：{_MODEL_PATH}")
        entry = raw[key]
        coef = {k: float(v) for k, v in entry["coef"].items()}
        models[key] = {
            "name": entry.get("name", key),
            "auc": entry.get("auc"),
            "coef": coef,
        }
        missing = set(FEATS) - set(coef)
        if missing:
            logger.warning("模型 %s 缺少特征系数 %s，将按 0 处理", key, sorted(missing))
    return models


_MODELS = _load_models()


def get_epi_week(ref_date: date | None = None) -> int:
    """Epidemiological week of the assessment date (1-52). Week 53 is folded into
    week 52, matching the 52-week encoding used during training."""
    ref = ref_date or date.today()
    return min(ref.isocalendar().week, 52)


def _answer_to_int(answer: str) -> int:
    """yes -> 1; no / unknown -> 0 (matching the SINAN feature engineering)."""
    return 1 if answer == "yes" else 0


def encode_features(form: FormInput, ref_date: date | None = None) -> MLFeatures:
    """Questionnaire answers -> the 26 model features (deterministic encoding, with no
    dependency on any external service).

    Note: form.exposure is **deliberately ignored** here. See the "Feature encoding"
    section of the module docstring.
    """
    values: dict[str, float | int] = {}
    for code in SYMPTOM_CODES:
        values[f"{code}_x"] = _answer_to_int(form.symptoms[code])
    for code in COMORB_CODES:
        values[f"{code}_x"] = _answer_to_int(form.comorbidities[code])

    week = get_epi_week(ref_date)
    values["age"] = float(form.age)
    values["sex_f"] = 1.0 if form.sex == "F" else 0.0
    values["day_ill"] = float(form.day_ill)
    values["wk_sin"] = math.sin(2 * math.pi * week / 52)
    values["wk_cos"] = math.cos(2 * math.pi * week / 52)
    return MLFeatures(**values)


def feature_code(name: str) -> str:
    """Feature name -> the label key the front end looks up.

    Symptoms and comorbidities drop the `_x` suffix (FEBRE_x -> FEBRE) so that the front
    end can reuse the multilingual labels the questionnaire already has; the 5
    non-binary features have no matching questionnaire item and keep their own names.
    """
    if name in NON_BINARY_FEATS:
        return name
    return name[:-2] if name.endswith("_x") else name


def _level(score: float) -> str:
    if score < _LOW_MAX:
        return "low"
    if score <= _MEDIUM_MAX:
        return "medium"
    return "high"


def score_from_z(coef: dict[str, float], z: float, wk_sin: float, wk_cos: float) -> float:
    """Linear predictor z -> 0-100 relative score (reference-person anchored, clipped,
    rounded to 1 decimal place).

    Both score_one and the planner's interval endpoints must go through this one
    function, so that the same z yields exactly the same score along every code path.
    """
    z_ref = _reference_z(coef, wk_sin, wk_cos)
    z_ceil = _ceiling_z(coef, wk_sin, wk_cos)
    span = z_ceil - z_ref
    ratio = (z - z_ref) / span if span > 0 else 0.0
    return round(100.0 * max(0.0, min(1.0, ratio)), 1)


class DengueModel:
    """Inference wrapper around the three logistic regression models."""

    def __init__(self, models: dict[str, dict] | None = None) -> None:
        self._models = models if models is not None else _MODELS

    def score_one(self, key: str, features: MLFeatures) -> ModelScore:
        """Score a single model.

        z = Σ coef × feature (no intercept);
        score = the position of z relative to the "same-season asymptomatic reference
        person", with the ceiling being every risk factor maxed out, clipped to
        [0, 100]. See the module docstring for details.
        """
        coef = self._models[key]["coef"]
        data = features.model_dump()
        z = sum(coef.get(name, 0.0) * float(data[name]) for name in FEATS)
        score = score_from_z(coef, z, data["wk_sin"], data["wk_cos"])
        return ModelScore(score=score, level=_level(score), z=round(z, 4))

    def score_all(self, features: MLFeatures) -> dict[str, ModelScore]:
        """Score all three models; returns {"A": …, "B": …, "B2": …}."""
        return {key: self.score_one(key, features) for key in MODEL_KEYS}

    def explain_one(
        self, key: str, features: MLFeatures, top_n: int = EXPLAIN_TOP_N
    ) -> list[FeatureContribution]:
        """Decompose a single model's z and return the largest contributions.

        Since z = Σ coef × feature, each term coef[name] × value is exactly how much
        that feature added to or subtracted from this score -- that is the whole of
        logistic regression interpretability, with no approximation involved.

        Terms contributing 0 (feature value 0, or a missing coefficient) are skipped:
        they have no effect on the result, and listing them would only dilute the few
        that actually matter. Sorted by absolute value, descending; the top top_n are
        returned.
        """
        coef = self._models[key]["coef"]
        data = features.model_dump()

        items: list[tuple[float, FeatureContribution]] = []
        for name in FEATS:
            contribution = coef.get(name, 0.0) * float(data[name])
            if contribution == 0.0:
                continue
            items.append(
                (
                    abs(contribution),
                    FeatureContribution(
                        feature=name,
                        code=feature_code(name),
                        contribution=round(contribution, 4),
                        direction="up" if contribution > 0 else "down",
                    ),
                )
            )

        items.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in items[:top_n]]

    def explain_all(
        self, features: MLFeatures, top_n: int = EXPLAIN_TOP_N
    ) -> dict[str, list[FeatureContribution]]:
        """Contribution breakdown for all three models, keyed by the response field
        names dengue / worsening / severe."""
        return {
            RESULT_FIELDS[key]: self.explain_one(key, features, top_n)
            for key in MODEL_KEYS
        }

    def coefficients(self, key: str) -> dict[str, float]:
        """Coefficient dictionary for one model (read-only; used by the planner to
        compute score bounds and information value)."""
        return self._models[key]["coef"]

    def info(self) -> dict[str, dict]:
        """Model metadata (name and AUC), for display by /api/health and similar."""
        return {
            key: {"name": m["name"], "auc": m["auc"]}
            for key, m in self._models.items()
        }


_model: DengueModel | None = None


def get_model() -> DengueModel:
    """Process-wide singleton."""
    global _model
    if _model is None:
        _model = DengueModel()
        logger.info("登革热模型已加载：%s", list(_model.info()))
    return _model
