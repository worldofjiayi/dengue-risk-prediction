"""Adaptive questioning planner: decides what to ask next, and proves when it is safe to stop.

== How it works (fully deterministic, no LLM) ==

The coefficients of the three logistic regression models are completely known, so for a
**partially answered** questionnaire the final score of each model can be given **hard
bounds**:

- the contribution of an answered binary question is certain: yes -> coef,
  no / unknown -> 0 (matching the SINAN feature engineering, see the ml_model module
  docstring);
- a binary question that has **not been asked yet** can only ever contribute 0 or its own
  coefficient, hence

      z_min = z_answered + Σ min(0, c_f)
      z_max = z_answered + Σ max(0, c_f)        (f ranges over the unasked features)

- age / sex / day_ill are required in the first step of the questionnaire, and the seasonal
  terms are computed by the server from the current date, so neither adds any uncertainty;
- feeding z_min / z_max through **exactly the same** reference-person normalisation as
  score_one (ml_model.score_from_z, clipped to [0, 100]) gives [score_min, score_max].

If all three models' intervals fall inside one risk band
(_level(score_min) == _level(score_max)), then **however they are answered** the remaining
questions cannot change any model's band -- a provably safe stop.

The key distinction: a user answering "don't know" is a certain 0 (answered); only "not yet
asked" is uncertain. PlanRequest tells the two states apart by whether the key is present
(see the notes on schemas.PlanRequest).

== Information value ==

The coefficients themselves define how much information each question carries. For an
unasked feature f:

    impact(f) = Σ_{undecided models m} |c_f^m| / (z_ceil^m − z_ref^m)

Dividing by each model's own normalisation span is what makes the three models'
coefficients comparable and addable. why_model takes the undecided model with the largest
normalised |coefficient| -- the front end uses it to explain "which estimate is this
question mainly helping to narrow". The ordering is fully determined: impact descending
first, ties broken by FEATS order (symptoms first, comorbidities after, each in the order
used by the training script).
"""

import math
from datetime import date

from app.ml_model import (
    MODEL_KEYS,
    RESULT_FIELDS,
    DengueModel,
    _ceiling_z,
    _level,
    _reference_z,
    get_epi_week,
    get_model,
    score_from_z,
)
from app.schemas import (
    COMORB_CODES,
    FEATS,
    SYMPTOM_CODES,
    ModelBounds,
    NextQuestion,
    PlanBounds,
    PlanRequest,
    PlanResponse,
)

# Every question the planner manages: (kind, code), in FEATS order
# (symptoms first, comorbidities after)
QUESTIONS: tuple[tuple[str, str], ...] = tuple(
    [("symptom", code) for code in SYMPTOM_CODES]
    + [("comorbidity", code) for code in COMORB_CODES]
)
QUESTION_COUNT = len(QUESTIONS)  # 21

# How many suggestions the next list returns at most
NEXT_MAX = 5


def _seasonal(ref_date: date | None) -> tuple[float, float]:
    """Seasonal terms for the current date (exactly the same formula as encode_features)."""
    week = get_epi_week(ref_date)
    return (
        math.sin(2 * math.pi * week / 52),
        math.cos(2 * math.pi * week / 52),
    )


def _answered_values(req: PlanRequest, wk_sin: float, wk_cos: float) -> dict[str, float]:
    """The 26 feature values known so far: yes -> 1, no / unknown / not asked -> 0.

    "Not asked counts as 0" is exactly the definition of score_now -- if the user stops
    answering right now, FormInput fills the missing keys in as unknown, they encode to 0,
    and this is the score that comes out.
    """
    values: dict[str, float] = {}
    for code in SYMPTOM_CODES:
        values[f"{code}_x"] = 1.0 if req.symptoms.get(code) == "yes" else 0.0
    for code in COMORB_CODES:
        values[f"{code}_x"] = 1.0 if req.comorbidities.get(code) == "yes" else 0.0
    values["age"] = float(req.age)
    values["sex_f"] = 1.0 if req.sex == "F" else 0.0
    values["day_ill"] = float(req.day_ill)
    values["wk_sin"] = wk_sin
    values["wk_cos"] = wk_cos
    return values


def _unasked_feats(req: PlanRequest) -> list[str]:
    """Binary features not yet asked (missing key = not asked; an answered unknown is not)."""
    feats = [f"{c}_x" for c in SYMPTOM_CODES if c not in req.symptoms]
    feats += [f"{c}_x" for c in COMORB_CODES if c not in req.comorbidities]
    return feats


def _model_bounds(
    coef: dict[str, float],
    values: dict[str, float],
    unasked: list[str],
    wk_sin: float,
    wk_cos: float,
) -> ModelBounds:
    """Hard [score_min, score_max] bounds and the current score for one model.

    The summation order for z matches score_one term by term (iterating over FEATS), which
    keeps score_now identical to the /api/assess score even in the floating-point sense.
    """
    z_now = sum(coef.get(name, 0.0) * values[name] for name in FEATS)
    z_min = z_now + sum(min(0.0, coef.get(f, 0.0)) for f in unasked)
    z_max = z_now + sum(max(0.0, coef.get(f, 0.0)) for f in unasked)

    score_now = score_from_z(coef, z_now, wk_sin, wk_cos)
    score_min = score_from_z(coef, z_min, wk_sin, wk_cos)
    score_max = score_from_z(coef, z_max, wk_sin, wk_cos)
    return ModelBounds(
        score_now=score_now,
        score_min=score_min,
        score_max=score_max,
        level_now=_level(score_now),
        decided=_level(score_min) == _level(score_max),
    )


def _rank_next(
    model: DengueModel,
    req: PlanRequest,
    undecided: list[str],
    wk_sin: float,
    wk_cos: float,
) -> list[NextQuestion]:
    """Rank the unasked questions by information value and take the top NEXT_MAX.

    impact(f) = Σ_{undecided models} |coef| / normalisation span; a question with zero
    impact on every undecided model is skipped outright -- asking it cannot possibly change
    any band that is still open.
    """
    spans: dict[str, float] = {}
    for key in undecided:
        coef = model.coefficients(key)
        spans[key] = _ceiling_z(coef, wk_sin, wk_cos) - _reference_z(coef, wk_sin, wk_cos)

    ranked: list[tuple[float, int, NextQuestion]] = []
    for index, (kind, code) in enumerate(QUESTIONS):
        answered = code in (req.symptoms if kind == "symptom" else req.comorbidities)
        if answered:
            continue
        feat = f"{code}_x"
        impact = 0.0
        best_key: str | None = None
        best_value = 0.0
        for key in undecided:  # MODEL_KEYS order; on a tie why_model takes the earlier model
            span = spans[key]
            if span <= 0:
                continue
            value = abs(model.coefficients(key).get(feat, 0.0)) / span
            impact += value
            if value > best_value:
                best_value = value
                best_key = key
        if best_key is None or impact <= 0.0:
            continue
        ranked.append(
            (
                -impact,
                index,
                NextQuestion(kind=kind, code=code, why_model=RESULT_FIELDS[best_key]),
            )
        )

    # Deterministic ordering: impact descending, ties by FEATS order
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [question for _, _, question in ranked[:NEXT_MAX]]


def plan(
    req: PlanRequest,
    ref_date: date | None = None,
    model: DengueModel | None = None,
) -> PlanResponse:
    """Core entry point: bounds -> can we stop -> next questions. Pure, no side effects.

    ref_date / model exist only for test injection; the production path uses the current
    date and the in-process singleton model.
    """
    model = model if model is not None else get_model()
    wk_sin, wk_cos = _seasonal(ref_date)
    values = _answered_values(req, wk_sin, wk_cos)
    unasked = _unasked_feats(req)

    bounds: dict[str, ModelBounds] = {}
    undecided: list[str] = []
    for key in MODEL_KEYS:
        block = _model_bounds(model.coefficients(key), values, unasked, wk_sin, wk_cos)
        bounds[RESULT_FIELDS[key]] = block
        if not block.decided:
            undecided.append(key)

    can_stop = not undecided
    answered = len(req.symptoms) + len(req.comorbidities)
    return PlanResponse(
        bounds=PlanBounds(**bounds),
        can_stop=can_stop,
        # Once every band is settled, no question is worth asking -- however many remain
        next=[] if can_stop else _rank_next(model, req, undecided, wk_sin, wk_cos),
        answered=answered,
        remaining=QUESTION_COUNT - answered,
    )
