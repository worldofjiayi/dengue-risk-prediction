"""自适应问诊规划器 /api/plan 的测试（纯确定性计算，无网络请求）。"""

import json
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# 与 app.schemas 保持一致的问题清单（在测试里显式写出，防止契约悄悄漂移）
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
    """强制 MOCK_MODE=true 并重置配置缓存后，构造 TestClient。"""
    monkeypatch.setenv("MOCK_MODE", "true")
    from app.config import get_settings

    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


def plan_body(**overrides) -> dict:
    """最小合法请求：只有必填的 age / sex / day_ill。"""
    body = {"age": 35, "sex": "F", "day_ill": 3}
    body.update(overrides)
    return body


def load_coefs() -> dict[str, dict[str, float]]:
    raw = json.loads(
        (ROOT / "app" / "model" / "dengue_models.json").read_text(encoding="utf-8")
    )
    return {key: raw[key]["coef"] for key in ("A", "B", "B2")}


def hand_span(coef: dict[str, float]) -> float:
    """独立实现的归一化跨度 z_ceil − z_ref（季节项在两端相同，天然抵消）。

    z_ceil：所有升高风险的特征取上界（age 110、day_ill 14、二值取 1）；
    z_ref ：30 岁男性、无症状、病程 0 天 —— 只剩 age 项。
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


# ---------- 空问卷：区间大开，next 按信息价值排序 ----------


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
        # 21 道题全部悬空时，区间必须是「大开」的，横跨不止一个档位
        assert block["score_max"] - block["score_min"] > 30.0

    assert 1 <= len(body["next"]) <= 5


def test_first_suggestion_has_highest_normalised_impact(client):
    """next[0] 必须是未定模型上归一化 |系数| 之和最大的问题。

    期望值在测试里用 JSON 系数独立算出，不照抄实现。
    """
    body = client.post("/api/plan", json=plan_body()).json()
    coefs = load_coefs()
    spans = {key: hand_span(coefs[key]) for key in coefs}
    undecided = [KEY_OF[f] for f in FIELDS if not body["bounds"][f]["decided"]]
    assert undecided, "空问卷至少应有一个模型未定档"

    def impact(code: str) -> float:
        return sum(
            abs(coefs[key].get(f"{code}_x", 0.0)) / spans[key] for key in undecided
        )

    all_codes = list(SYMPTOM_CODES + COMORB_CODES)
    expected_top = max(all_codes, key=impact)  # 平手取靠前者，与 FEATS 顺序一致
    got = body["next"][0]
    assert got["code"] == expected_top

    # why_model：该问题归一化 |系数| 最大的未定模型
    expected_why = max(
        undecided,
        key=lambda key: abs(coefs[key].get(f"{expected_top}_x", 0.0)) / spans[key],
    )
    assert got["why_model"] == FIELD_OF[expected_why]

    # 整个 next 列表按 impact 严格不升序排列，且只含未问的合法代码
    impacts = [impact(item["code"]) for item in body["next"]]
    assert impacts == sorted(impacts, reverse=True)
    for item in body["next"]:
        assert item["code"] in all_codes
        expected_kind = "symptom" if item["code"] in SYMPTOM_CODES else "comorbidity"
        assert item["kind"] == expected_kind


# ---------- 完整作答：与 /api/assess 完全一致 ----------

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
    """答完全部 21 题：区间收成一个点，且 score_now == /api/assess 的分数。"""
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
    """部分作答时 score_now == 同一份答案走 /api/assess 的分数。

    /api/assess 的 FormInput 会把缺失键补成 unknown（编码 0），
    这正是「未问按 0 计」的语义——两条路径必须给出同一个数。
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


# ---------- 「已答不知道」与「还没问」是两回事 ----------


def test_answered_unknown_tightens_exactly_like_no(client):
    """答「不知道」与答「否」必须给出完全相同的规划结果。"""
    with_unknown = client.post(
        "/api/plan", json=plan_body(comorbidities={"DIABETES": "unknown"})
    ).json()
    with_no = client.post(
        "/api/plan", json=plan_body(comorbidities={"DIABETES": "no"})
    ).json()
    assert with_unknown == with_no


def test_answered_unknown_tightens_versus_unasked(client):
    """答过「不知道」后区间必须收窄，且该题从 next 中消失。

    DIABETES 在三个模型中系数均为正（0.093 / 0.389 / 0.655），
    回答后 score_max 应严格下降，score_min 不动（正系数只影响上界）。
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
        assert after["score_now"] == before["score_now"]  # unknown 编码为 0，当前分不变

    assert "DIABETES" not in [item["code"] for item in answered["next"]]


# ---------- 单调收窄：作答永不扩大区间 ----------


def test_answering_never_widens_any_interval(client):
    """从空问卷出发，任答一题（任一答案）都不得扩大任何模型的区间。"""
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
    """从部分作答状态出发同样成立。"""
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


# ---------- 提前停止：整个规划器存在的意义 ----------


def test_early_stop_with_many_questions_unasked(client):
    """精心构造：25 岁男性、病程 0 天，对 7 道高影响题答「否」后，
    三个模型全部定档为 low —— 还剩 14 道题没问也可以停。
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
    assert body["remaining"] == 14  # 大量问题未问，但已可证明性地停止
    assert body["next"] == []
    for field in FIELDS:
        block = body["bounds"][field]
        assert block["decided"] is True
        assert block["level_now"] == "low"
        assert block["score_max"] < 35.0


def test_greedy_loop_following_planner_stops_early(client):
    """自适应闭环：每轮都答掉规划器的第一条建议（答「否」），
    必须在问完 21 题之前就到达 can_stop。
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
    assert body["remaining"] > 0, "跟着规划器走必须能在问完之前停下"
    assert body["next"] == []
    # 高影响题优先意味着收敛应当很快（人工推演为 7 题左右）
    assert body["answered"] <= 10


# ---------- decided 必须严格尊重 35 / 65 的分档边界 ----------


def _fake_model(coef: dict[str, float]):
    """三个键位使用同一份合成系数的注入模型。"""
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
    """[35.0, 65.0] 两端都在 medium（35 含、65 含）→ decided。

    合成系数总正量 = 1.0（归一化跨度为 1），FEBRE 已答 yes 贡献 0.35，
    MIALGIA 未问可再加 0.30，LEUCOPENIA 已答 no。
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
    assert result.remaining == 19  # 定档后 next 为空，与还剩多少题无关


def test_interval_crossing_35_is_not_decided():
    """[34.9, 65.0] 下端落进 low → 不能停。"""
    coef = {"FEBRE_x": 0.349, "MIALGIA_x": 0.301, "LEUCOPENIA_x": 0.35}
    result = _plan_direct(coef, {"FEBRE": "yes", "LEUCOPENIA": "no"})

    for field in FIELDS:
        block = getattr(result.bounds, field)
        assert block.score_min == 34.9
        assert block.score_max == 65.0
        assert block.decided is False
    assert result.can_stop is False
    # 唯一有非零影响的未问题是 MIALGIA；零系数问题不值得建议
    assert [item.code for item in result.next] == ["MIALGIA"]
    assert result.next[0].why_model == "dengue"  # 三模型同系数，平手取靠前者


def test_interval_crossing_65_is_not_decided():
    """[65.0, 65.1] 上端落进 high → 不能停。"""
    coef = {"FEBRE_x": 0.65, "MIALGIA_x": 0.001, "LEUCOPENIA_x": 0.349}
    result = _plan_direct(coef, {"FEBRE": "yes", "LEUCOPENIA": "no"})

    for field in FIELDS:
        block = getattr(result.bounds, field)
        assert block.score_min == 65.0
        assert block.score_max == 65.1
        assert block.decided is False
    assert result.can_stop is False


# ---------- 季节无关性 ----------


def test_bounds_independent_of_season():
    """季节项在 z、z_ref、z_ceil 中相同，归一化后抵消：换周次结果不变。"""
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


# ---------- 输入校验 ----------


def test_unknown_symptom_code_rejected(client):
    resp = client.post("/api/plan", json=plan_body(symptoms={"NOT_A_SYMPTOM": "yes"}))
    assert resp.status_code == 422


def test_symptom_code_in_comorbidities_rejected(client):
    """症状代码放错到 comorbidities 也必须 422，不能静默吞掉。"""
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
    """age / sex / day_ill 是规划的前提（必填第一步），缺一不可。"""
    for missing in ("age", "sex", "day_ill"):
        body = plan_body()
        del body[missing]
        assert client.post("/api/plan", json=body).status_code == 422


def test_missing_keys_stay_unasked_not_unknown():
    """PlanRequest 绝不能补全缺失键——缺失本身就是「未问」信号。"""
    from app.schemas import PlanRequest

    req = PlanRequest.model_validate(plan_body(symptoms={"FEBRE": "yes"}))
    assert req.symptoms == {"FEBRE": "yes"}          # 只保留真的答过的
    assert req.comorbidities == {}
    assert "LEUCOPENIA" not in req.symptoms          # 不像 FormInput 那样补 unknown
