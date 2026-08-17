"""Tests for the adaptive questioning planner /api/plan (deterministic, no network)."""

import json
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Question list kept in step with app.schemas (written out explicitly in the test,
# so the contract cannot drift silently)
SYMPTOM_CODES = (
    "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA",
    "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N",
    "LEUCOPENIA", "LACO", "DOR_RETRO",
)
COMORB_CODES = (
    "DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL",
    "HIPERTENSA", "ACIDO_PEPT", "AUTO_IMUNE",
)
FIELDS = ("dengue", "worsening", "severe")
KEY_OF = {"dengue": "A", "worsening": "B", "severe": "B2"}
FIELD_OF = {"A": "dengue", "B": "worsening", "B2": "severe"}


@pytest.fixture()
def client(monkeypatch):
    """Build a TestClient after forcing MOCK_MODE=true and clearing the settings cache."""
    monkeypatch.setenv("MOCK_MODE", "true")
    from app.config import get_settings

    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


def plan_body(**overrides) -> dict:
    """Minimal valid request: only the mandatory age / sex / day_ill."""
    body = {"age": 35, "sex": "F", "day_ill": 3}
    body.update(overrides)
    return body


def load_coefs() -> dict[str, dict[str, float]]:
    raw = json.loads(
        (ROOT / "app" / "model" / "dengue_models.json").read_text(encoding="utf-8")
    )
    return {key: raw[key]["coef"] for key in ("A", "B", "B2")}


def hand_span(coef: dict[str, float]) -> float:
    """Independently implemented normalisation span z_ceil - z_ref.

    The seasonal terms are the same at both ends, so they cancel naturally.
    z_ceil: every risk-raising feature at its upper bound (age 110, day_ill 14, binary 1);
    z_ref : a 30-year-old male, no symptoms, 0 days of illness -- only the age term left.
    """
    total = 0.0
    for name, c in coef.items():
        if name in ("wk_sin", "wk_cos"):
            continue
        if name == "age":
            total += max(0.0, c * 110.0)
        elif name == "day_ill":
            total += max(0.0, c * 14.0)
        else:
            total += max(0.0, c)
    return total - coef["age"] * 30.0


# ---------- Empty questionnaire: wide intervals, next sorted by information value ----------


def test_empty_answers_wide_intervals_cannot_stop(client):
    resp = client.post("/api/plan", json=plan_body())
    assert resp.status_code == 200
    body = resp.json()

    assert body["answered"] == 0
    assert body["remaining"] == 21
    assert body["can_stop"] is False

    for field in FIELDS:
        block = body["bounds"][field]
        assert 0.0 <= block["score_min"] <= block["score_now"] <= block["score_max"] <= 100.0
        assert block["decided"] is False
        # With all 21 questions open the interval must be wide, spanning more than one band
        assert block["score_max"] - block["score_min"] > 30.0

    assert 1 <= len(body["next"]) <= 5


def test_first_suggestion_has_highest_normalised_impact(client):
    """next[0] must maximise the summed normalised |coefficient| over undecided models.

    The expected value is computed in the test from the JSON coefficients rather than
    copied from the implementation.
    """
    body = client.post("/api/plan", json=plan_body()).json()
    coefs = load_coefs()
    spans = {key: hand_span(coefs[key]) for key in coefs}
    undecided = [KEY_OF[f] for f in FIELDS if not body["bounds"][f]["decided"]]
    assert undecided, "an empty questionnaire must leave at least one model undecided"

    def impact(code: str) -> float:
        return sum(
            abs(coefs[key].get(f"{code}_x", 0.0)) / spans[key] for key in undecided
        )

    all_codes = list(SYMPTOM_CODES + COMORB_CODES)
    expected_top = max(all_codes, key=impact)  # ties go to the earlier one, as in FEATS order
    got = body["next"][0]
    assert got["code"] == expected_top

    # why_model: the undecided model with this question's largest normalised |coefficient|
    expected_why = max(
        undecided,
        key=lambda key: abs(coefs[key].get(f"{expected_top}_x", 0.0)) / spans[key],
    )
    assert got["why_model"] == FIELD_OF[expected_why]

    # The whole next list is non-increasing in impact, and holds only unasked valid codes
    impacts = [impact(item["code"]) for item in body["next"]]
    assert impacts == sorted(impacts, reverse=True)
    for item in body["next"]:
        assert item["code"] in all_codes
        expected_kind = "symptom" if item["code"] in SYMPTOM_CODES else "comorbidity"
        assert item["kind"] == expected_kind


# ---------- Fully answered: identical to /api/assess ----------

FULL_SYMPTOMS = {
    "FEBRE": "yes", "MIALGIA": "yes", "CEFALEIA": "no", "EXANTEMA": "no",
    "VOMITO": "no", "NAUSEA": "yes", "DOR_COSTAS": "no", "CONJUNTVIT": "no",
    "ARTRITE": "no", "ARTRALGIA": "unknown", "PETEQUIA_N": "no",
    "LEUCOPENIA": "unknown", "LACO": "unknown", "DOR_RETRO": "yes",
}
FULL_COMORB = {
    "DIABETES": "yes", "HEMATOLOG": "no", "HEPATOPAT": "no", "RENAL": "no",
    "HIPERTENSA": "unknown", "ACIDO_PEPT": "no", "AUTO_IMUNE": "no",
}


def test_fully_answered_collapses_and_matches_assess(client):
    """All 21 answered: the interval collapses to a point and score_now == /api/assess."""
    body = client.post(
        "/api/plan",
        json=plan_body(symptoms=FULL_SYMPTOMS, comorbidities=FULL_COMORB),
    ).json()

    assert body["answered"] == 21
    assert body["remaining"] == 0
    assert body["can_stop"] is True
    assert body["next"] == []

    assess = client.post(
        "/api/assess",
        json={
            "age": 35, "sex": "F", "day_ill": 3,
            "symptoms": FULL_SYMPTOMS, "comorbidities": FULL_COMORB,
        },
    ).json()

    for field in FIELDS:
        block = body["bounds"][field]
        assert block["decided"] is True
        assert block["score_min"] == block["score_now"] == block["score_max"]
        assert block["score_now"] == assess[field]["score"]
        assert block["level_now"] == assess[field]["level"]


def test_partial_score_now_matches_assess_with_missing_as_unknown(client):
    """Partly answered: score_now == the score the same answers get from /api/assess.

    FormInput in /api/assess fills missing keys in as unknown (encoded 0), which is
    exactly the "unasked counts as 0" semantics -- both paths must give the same number.
    """
    symptoms = {"FEBRE": "yes", "VOMITO": "no", "LEUCOPENIA": "unknown"}
    comorb = {"DIABETES": "unknown"}

    body = client.post(
        "/api/plan", json=plan_body(symptoms=symptoms, comorbidities=comorb)
    ).json()
    assess = client.post(
        "/api/assess",
        json={"age": 35, "sex": "F", "day_ill": 3,
              "symptoms": symptoms, "comorbidities": comorb},
    ).json()

    assert body["answered"] == 4
    assert body["remaining"] == 17
    for field in FIELDS:
        assert body["bounds"][field]["score_now"] == assess[field]["score"]
        assert body["bounds"][field]["level_now"] == assess[field]["level"]


# ---------- "Answered do not know" and "not yet asked" are different ----------


def test_answered_unknown_tightens_exactly_like_no(client):
    """Answering "do not know" and answering "no" must give identical planning results."""
    with_unknown = client.post(
        "/api/plan", json=plan_body(comorbidities={"DIABETES": "unknown"})
    ).json()
    with_no = client.post(
        "/api/plan", json=plan_body(comorbidities={"DIABETES": "no"})
    ).json()
    assert with_unknown == with_no


def test_answered_unknown_tightens_versus_unasked(client):
    """Once "do not know" is answered the interval must narrow and the question leave next.

    DIABETES has a positive coefficient in all three models (0.093 / 0.389 / 0.655), so
    after answering score_max must fall strictly while score_min stays put (a positive
    coefficient only moves the upper bound).
    """
    unasked = client.post("/api/plan", json=plan_body()).json()
    answered = client.post(
        "/api/plan", json=plan_body(comorbidities={"DIABETES": "unknown"})
    ).json()

    assert answered["answered"] == 1
    assert answered["remaining"] == 20
    for field in FIELDS:
        before, after = unasked["bounds"][field], answered["bounds"][field]
        assert after["score_max"] < before["score_max"]
        assert after["score_min"] == before["score_min"]
        assert after["score_now"] == before["score_now"]  # unknown encodes 0, so no change

    assert "DIABETES" not in [item["code"] for item in answered["next"]]


# ---------- Monotone narrowing: answering never widens an interval ----------


def test_answering_never_widens_any_interval(client):
    """From an empty questionnaire, any single answer must not widen any model's interval."""
    base = client.post("/api/plan", json=plan_body()).json()

    for kind_field, codes in (("symptoms", SYMPTOM_CODES), ("comorbidities", COMORB_CODES)):
        for code in codes:
            for answer in ("yes", "no", "unknown"):
                after = client.post(
                    "/api/plan", json=plan_body(**{kind_field: {code: answer}})
                ).json()
                for field in FIELDS:
                    b0, b1 = base["bounds"][field], after["bounds"][field]
                    assert b1["score_min"] >= b0["score_min"], (code, answer, field)
                    assert b1["score_max"] <= b0["score_max"], (code, answer, field)


def test_answering_never_widens_from_partial_state(client):
    """The same holds starting from a partly answered state."""
    answered_symptoms = {"FEBRE": "yes", "LEUCOPENIA": "no"}
    answered_comorb = {"RENAL": "unknown"}
    base = client.post(
        "/api/plan",
        json=plan_body(symptoms=answered_symptoms, comorbidities=answered_comorb),
    ).json()

    for code in SYMPTOM_CODES:
        if code in answered_symptoms:
            continue
        for answer in ("yes", "no"):
            after = client.post(
                "/api/plan",
                json=plan_body(
                    symptoms={**answered_symptoms, code: answer},
                    comorbidities=answered_comorb,
                ),
            ).json()
            for field in FIELDS:
                b0, b1 = base["bounds"][field], after["bounds"][field]
                assert b1["score_min"] >= b0["score_min"], (code, answer, field)
                assert b1["score_max"] <= b0["score_max"], (code, answer, field)


# ---------- Early stopping: the whole point of the planner ----------


def test_early_stop_with_many_questions_unasked(client):
    """Carefully built: a 25-year-old male, 0 days of illness, answering "no" to the 7
    highest-impact questions puts all three models in the low band -- 14 questions can
    stay unasked and it can still stop.
    """
    body = client.post(
        "/api/plan",
        json={
            "age": 25, "sex": "M", "day_ill": 0,
            "symptoms": {
                "FEBRE": "no", "MIALGIA": "no", "LEUCOPENIA": "no", "VOMITO": "no",
            },
            "comorbidities": {
                "DIABETES": "no", "RENAL": "no", "AUTO_IMUNE": "no",
            },
        },
    ).json()

    assert body["can_stop"] is True
    assert body["answered"] == 7
    assert body["remaining"] == 14  # many questions unasked, but stopping is now provable
    assert body["next"] == []
    for field in FIELDS:
        block = body["bounds"][field]
        assert block["decided"] is True
        assert block["level_now"] == "low"
        assert block["score_max"] < 35.0


def test_greedy_loop_following_planner_stops_early(client):
    """Adaptive loop: answer the planner's first suggestion each round (with "no"), and
    can_stop must be reached before all 21 questions have been asked.
    """
    symptoms: dict[str, str] = {}
    comorbidities: dict[str, str] = {}
    body = None
    for _ in range(21):
        body = client.post(
            "/api/plan",
            json={"age": 25, "sex": "M", "day_ill": 0,
                  "symptoms": symptoms, "comorbidities": comorbidities},
        ).json()
        if body["can_stop"]:
            break
        top = body["next"][0]
        target = symptoms if top["kind"] == "symptom" else comorbidities
        target[top["code"]] = "no"

    assert body is not None and body["can_stop"] is True
    assert body["remaining"] > 0, "following the planner must stop before the list runs out"
    assert body["next"] == []
    # High-impact questions first means fast convergence (hand-traced at about 7 questions)
    assert body["answered"] <= 10


# ---------- decided must strictly respect the 35 / 65 band boundaries ----------


def _fake_model(coef: dict[str, float]):
    """An injected model whose three keys all share the same synthetic coefficients."""
    from app.ml_model import DengueModel

    models = {
        key: {"name": key, "auc": None, "coef": dict(coef)}
        for key in ("A", "B", "B2")
    }
    return DengueModel(models=models)


def _plan_direct(coef: dict[str, float], symptoms: dict[str, str]):
    from app.planner import plan
    from app.schemas import PlanRequest

    req = PlanRequest(age=30, sex="M", day_ill=0, symptoms=symptoms)
    return plan(req, ref_date=date(2026, 8, 16), model=_fake_model(coef))


def test_interval_exactly_35_to_65_is_decided_medium():
    """[35.0, 65.0] has both ends in medium (35 included, 65 included) -> decided.

    Total positive synthetic coefficient = 1.0 (normalisation span 1); FEBRE answered
    yes contributes 0.35, unasked MIALGIA could add 0.30, LEUCOPENIA answered no.
    """
    coef = {"FEBRE_x": 0.35, "MIALGIA_x": 0.30, "LEUCOPENIA_x": 0.35}
    result = _plan_direct(coef, {"FEBRE": "yes", "LEUCOPENIA": "no"})

    for field in FIELDS:
        block = getattr(result.bounds, field)
        assert block.score_min == 35.0
        assert block.score_max == 65.0
        assert block.decided is True
    assert result.can_stop is True
    assert result.next == []
    assert result.remaining == 19  # next empties once decided, regardless of how many remain


def test_interval_crossing_35_is_not_decided():
    """[34.9, 65.0] has its lower end in low -> cannot stop."""
    coef = {"FEBRE_x": 0.349, "MIALGIA_x": 0.301, "LEUCOPENIA_x": 0.35}
    result = _plan_direct(coef, {"FEBRE": "yes", "LEUCOPENIA": "no"})

    for field in FIELDS:
        block = getattr(result.bounds, field)
        assert block.score_min == 34.9
        assert block.score_max == 65.0
        assert block.decided is False
    assert result.can_stop is False
    # MIALGIA is the only unasked question with non-zero impact; a zero-coefficient
    # question is not worth suggesting
    assert [item.code for item in result.next] == ["MIALGIA"]
    assert result.next[0].why_model == "dengue"  # equal coefficients: ties go to the first


def test_interval_crossing_65_is_not_decided():
    """[65.0, 65.1] has its upper end in high -> cannot stop."""
    coef = {"FEBRE_x": 0.65, "MIALGIA_x": 0.001, "LEUCOPENIA_x": 0.349}
    result = _plan_direct(coef, {"FEBRE": "yes", "LEUCOPENIA": "no"})

    for field in FIELDS:
        block = getattr(result.bounds, field)
        assert block.score_min == 65.0
        assert block.score_max == 65.1
        assert block.decided is False
    assert result.can_stop is False


# ---------- Season independence ----------


def test_bounds_independent_of_season():
    """Seasonal terms are equal in z, z_ref and z_ceil and cancel: the week does not matter."""
    from app.planner import plan
    from app.schemas import PlanRequest

    req = PlanRequest(
        age=40, sex="F", day_ill=5,
        symptoms={"FEBRE": "yes", "LEUCOPENIA": "unknown"},
    )
    january = plan(req, ref_date=date(2026, 1, 15))
    july = plan(req, ref_date=date(2026, 7, 15))
    assert january.bounds == july.bounds
    assert january.next == july.next
    assert january.can_stop == july.can_stop


# ---------- Input validation ----------


def test_unknown_symptom_code_rejected(client):
    resp = client.post("/api/plan", json=plan_body(symptoms={"NOT_A_SYMPTOM": "yes"}))
    assert resp.status_code == 422


def test_symptom_code_in_comorbidities_rejected(client):
    """A symptom code misplaced into comorbidities must 422, not be swallowed silently."""
    resp = client.post("/api/plan", json=plan_body(comorbidities={"FEBRE": "yes"}))
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"age": 200},
        {"age": -1},
        {"day_ill": 15},
        {"day_ill": -1},
        {"sex": "X"},
        {"language": "fr"},
        {"symptoms": {"FEBRE": "maybe"}},
    ],
)
def test_out_of_range_values_rejected(client, overrides):
    resp = client.post("/api/plan", json=plan_body(**overrides))
    assert resp.status_code == 422


def test_mandatory_first_step_fields_required(client):
    """age / sex / day_ill are the mandatory first step of planning; none may be missing."""
    for missing in ("age", "sex", "day_ill"):
        body = plan_body()
        del body[missing]
        assert client.post("/api/plan", json=body).status_code == 422


def test_missing_keys_stay_unasked_not_unknown():
    """PlanRequest must never fill in missing keys -- absence is the "unasked" signal."""
    from app.schemas import PlanRequest

    req = PlanRequest.model_validate(plan_body(symptoms={"FEBRE": "yes"}))
    assert req.symptoms == {"FEBRE": "yes"}          # only what was really answered
    assert req.comorbidities == {}
    assert "LEUCOPENIA" not in req.symptoms          # no unknown filled in, unlike FormInput
