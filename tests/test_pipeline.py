"""登革热风险评估流水线与 API 集成测试（全部在 MOCK_MODE 下运行，不发真实网络请求）。"""

import json
import math
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


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


def make_form(**overrides) -> dict:
    """构造一份合法问卷，可用关键字覆盖任意字段。"""
    form = {
        "age": 34,
        "sex": "F",
        "day_ill": 3,
        "symptoms": {
            "FEBRE": "yes",
            "MIALGIA": "yes",
            "CEFALEIA": "yes",
            "EXANTEMA": "no",
            "VOMITO": "no",
            "NAUSEA": "yes",
            "DOR_COSTAS": "no",
            "CONJUNTVIT": "no",
            "ARTRITE": "no",
            "ARTRALGIA": "yes",
            "PETEQUIA_N": "no",
            "LEUCOPENIA": "unknown",
            "LACO": "unknown",
            "DOR_RETRO": "yes",
        },
        "comorbidities": {
            "DIABETES": "no",
            "HEMATOLOG": "no",
            "HEPATOPAT": "no",
            "RENAL": "no",
            "HIPERTENSA": "no",
            "ACIDO_PEPT": "no",
            "AUTO_IMUNE": "no",
        },
    }
    form.update(overrides)
    return form


ALL_NO_SYMPTOMS = {
    c: "no"
    for c in (
        "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA",
        "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N",
        "LEUCOPENIA", "LACO", "DOR_RETRO",
    )
}
ALL_NO_COMORB = {
    c: "no"
    for c in (
        "DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL",
        "HIPERTENSA", "ACIDO_PEPT", "AUTO_IMUNE",
    )
}


# ---------- API 基本行为 ----------


def test_health_reports_three_models(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mock_mode"] is True
    assert set(body["models"]) == {"A", "B", "B2"}


def test_assess_returns_three_scores(client):
    resp = client.post("/api/assess", json=make_form())
    assert resp.status_code == 200
    body = resp.json()

    for field in ("dengue", "worsening", "severe"):
        block = body[field]
        assert 0.0 <= block["score"] <= 100.0
        assert block["level"] in {"low", "medium", "high"}
        assert isinstance(block["z"], float)

    assert 1 <= body["epi_week"] <= 52
    assert body["summary"]
    assert body["disclaimer"]
    assert body["model_note"]
    for key in ("protection", "medical", "monitoring"):
        assert len(body["advice"][key]) > 0


def test_missing_symptom_keys_default_to_unknown(client):
    """只填了部分症状也能评估，缺失项按 unknown 处理。"""
    resp = client.post(
        "/api/assess",
        json=make_form(symptoms={"FEBRE": "yes"}, comorbidities={}),
    )
    assert resp.status_code == 200


def test_unknown_symptom_key_rejected(client):
    resp = client.post(
        "/api/assess", json=make_form(symptoms={"NOT_A_SYMPTOM": "yes"})
    )
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
    ],
)
def test_out_of_range_values_rejected(client, overrides):
    resp = client.post("/api/assess", json=make_form(**overrides))
    assert resp.status_code == 422


# ---------- 五语言 ----------

DISCLAIMER_TEXTS = {
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


@pytest.mark.parametrize("language", sorted(DISCLAIMER_TEXTS))
def test_each_language(client, language):
    resp = client.post("/api/assess", json=make_form(language=language))
    assert resp.status_code == 200
    body = resp.json()
    assert body["disclaimer"] == DISCLAIMER_TEXTS[language]
    assert body["model_note"]
    assert body["summary"]


def test_language_defaults_to_zh_cn(client):
    form = make_form()
    assert "language" not in form
    resp = client.post("/api/assess", json=form)
    assert resp.status_code == 200
    assert resp.json()["disclaimer"] == DISCLAIMER_TEXTS["zh-CN"]


def test_model_note_never_claims_probability(client):
    """model_note 必须说明这是相对评分，不能被误读为概率。"""
    resp = client.post("/api/assess", json=make_form(language="en"))
    note = resp.json()["model_note"]
    assert "relative risk" in note.lower()
    assert "not infection probabilities" in note.lower()


# ---------- 特征编码 ----------


def test_unknown_encodes_same_as_no():
    """SINAN 特征工程里 9=未知 与 2=无 都记为 0，编码必须一致。"""
    from app.ml_model import encode_features
    from app.schemas import FormInput

    ref = date(2026, 8, 16)
    unknown_form = FormInput(
        **make_form(
            symptoms={**ALL_NO_SYMPTOMS, "LEUCOPENIA": "unknown", "LACO": "unknown"},
            comorbidities=ALL_NO_COMORB,
        )
    )
    no_form = FormInput(
        **make_form(
            symptoms={**ALL_NO_SYMPTOMS, "LEUCOPENIA": "no", "LACO": "no"},
            comorbidities=ALL_NO_COMORB,
        )
    )
    assert (
        encode_features(unknown_form, ref).model_dump()
        == encode_features(no_form, ref).model_dump()
    )


def test_sex_and_seasonal_encoding():
    from app.ml_model import encode_features, get_epi_week
    from app.schemas import FormInput

    ref = date(2026, 8, 16)
    week = get_epi_week(ref)
    assert 1 <= week <= 52

    female = encode_features(FormInput(**make_form(sex="F")), ref)
    male = encode_features(FormInput(**make_form(sex="M")), ref)
    assert female.sex_f == 1.0
    assert male.sex_f == 0.0
    assert female.wk_sin == pytest.approx(math.sin(2 * math.pi * week / 52))
    assert female.wk_cos == pytest.approx(math.cos(2 * math.pi * week / 52))


def test_feature_vector_matches_training_order():
    """as_vector 必须按训练脚本的 FEATS 顺序展开，共 26 维。"""
    from app.ml_model import encode_features
    from app.schemas import FEATS, FormInput

    features = encode_features(FormInput(**make_form()), date(2026, 8, 16))
    vector = features.as_vector()
    assert len(vector) == 26
    assert len(FEATS) == 26
    assert FEATS[0] == "FEBRE_x"
    assert FEATS[13] == "DOR_RETRO_x"
    assert FEATS[14] == "DIABETES_x"
    assert FEATS[-5:] == ("age", "sex_f", "day_ill", "wk_sin", "wk_cos")


# ---------- 模型打分正确性（关键） ----------


def test_z_matches_hand_computed_value():
    """独立手算 z，验证系数读取与点乘没有错位。

    构造：只有发热，年龄 0，男性，病程 0 天 —— 只剩 FEBRE 与季节项贡献。
    模型 A 的系数（取自 app/model/dengue_models.json）：
        FEBRE_x = 0.904, wk_sin = 0.432, wk_cos = -0.249
    """
    from app.ml_model import DengueModel, encode_features, get_epi_week
    from app.schemas import FormInput

    ref = date(2026, 8, 16)
    week = get_epi_week(ref)
    form = FormInput(
        **make_form(
            age=0,
            sex="M",
            day_ill=0,
            symptoms={**ALL_NO_SYMPTOMS, "FEBRE": "yes"},
            comorbidities=ALL_NO_COMORB,
        )
    )
    features = encode_features(form, ref)

    expected_z = (
        0.904
        + 0.432 * math.sin(2 * math.pi * week / 52)
        + (-0.249) * math.cos(2 * math.pi * week / 52)
    )
    model = DengueModel()
    got = model.score_one("A", features)
    assert got.z == pytest.approx(expected_z, abs=1e-3)

    # 分数 = z 相对「同季节无症状参考人」的位置
    from app.ml_model import _ceiling_z, _reference_z

    coef = model._models["A"]["coef"]
    wk_sin, wk_cos = features.wk_sin, features.wk_cos
    z_ref = _reference_z(coef, wk_sin, wk_cos)
    z_ceil = _ceiling_z(coef, wk_sin, wk_cos)
    expected_score = 100.0 * (expected_z - z_ref) / (z_ceil - z_ref)
    assert got.score == pytest.approx(expected_score, abs=0.1)


def test_reference_person_scores_zero_and_worst_case_scores_100():
    """无症状参考人应为 0 分，风险因子拉满应为 100 分。"""
    from app.ml_model import DengueModel, _REF_AGE
    from app.schemas import FEATS, MLFeatures

    model = DengueModel()
    week_feats = {"wk_sin": 0.3, "wk_cos": -0.5}

    for key in ("A", "B", "B2"):
        coef = model._models[key]["coef"]

        ref = {name: 0.0 for name in FEATS}
        ref.update(week_feats)
        ref["age"] = _REF_AGE
        assert model.score_one(key, MLFeatures(**ref)).score == 0.0

        worst = {}
        for name in FEATS:
            if name in week_feats:
                worst[name] = week_feats[name]
            elif name == "age":
                worst[name] = 110.0 if coef.get(name, 0.0) > 0 else 0.0
            elif name == "day_ill":
                worst[name] = 14.0 if coef.get(name, 0.0) > 0 else 0.0
            else:
                worst[name] = 1 if coef.get(name, 0.0) > 0 else 0
        assert model.score_one(key, MLFeatures(**worst)).score == 100.0


def test_season_cancels_out_of_the_score():
    """季节项在归一化中抵消：换一个周次，z 变化但分数不变。"""
    from app.ml_model import DengueModel
    from app.schemas import FEATS, MLFeatures

    model = DengueModel()
    base = {name: 0.0 for name in FEATS}
    base.update({"age": 45.0, "day_ill": 4.0, "FEBRE_x": 1, "LEUCOPENIA_x": 1})

    winter = MLFeatures(**{**base, "wk_sin": 0.9, "wk_cos": 0.2})
    summer = MLFeatures(**{**base, "wk_sin": -0.7, "wk_cos": -0.6})

    for key in ("A", "B", "B2"):
        a, b = model.score_one(key, winter), model.score_one(key, summer)
        assert a.z != b.z                      # 线性预测值确实随季节变化
        assert a.score == pytest.approx(b.score, abs=0.05)  # 但相对分数不变


def test_typical_cases_spread_across_levels():
    """健康年轻人应为低风险；典型登革热病例不应被一律判成高风险。"""
    from app.ml_model import DengueModel, encode_features
    from app.schemas import FormInput

    ref = date(2026, 8, 16)
    model = DengueModel()

    healthy = FormInput(
        **make_form(age=25, sex="M", day_ill=0,
                    symptoms=ALL_NO_SYMPTOMS, comorbidities=ALL_NO_COMORB)
    )
    typical = FormInput(
        **make_form(age=35, sex="F", day_ill=3,
                    symptoms={**ALL_NO_SYMPTOMS, "FEBRE": "yes", "CEFALEIA": "yes",
                              "MIALGIA": "yes", "DOR_RETRO": "yes"},
                    comorbidities=ALL_NO_COMORB)
    )

    low = model.score_all(encode_features(healthy, ref))
    mid = model.score_all(encode_features(typical, ref))

    assert low["A"].level == "low"
    assert low["B2"].level == "low"
    # 典型病例不应被判成重症高风险（无合并症、无白细胞减少）
    assert mid["B2"].level in {"low", "medium"}
    assert mid["A"].score > low["A"].score


def test_coefficients_loaded_from_bundled_json():
    """模型系数必须来自随包的 JSON 文件，而不是硬编码在代码里。"""
    from app.ml_model import DengueModel

    raw = json.loads(
        (ROOT / "app" / "model" / "dengue_models.json").read_text(encoding="utf-8")
    )
    model = DengueModel()
    info = model.info()
    assert set(info) == {"A", "B", "B2"}
    # AUC 与文件一致，说明确实读的是这份文件
    assert info["B2"]["auc"] == raw["B2"]["auc"] == 0.8096


def test_more_symptoms_scores_higher():
    """症状与合并症越多，登革热与重症评分越高。"""
    from app.ml_model import DengueModel, encode_features
    from app.schemas import FormInput

    ref = date(2026, 8, 16)
    model = DengueModel()

    healthy = FormInput(
        **make_form(
            age=25, sex="M", day_ill=0,
            symptoms=ALL_NO_SYMPTOMS, comorbidities=ALL_NO_COMORB,
        )
    )
    sick = FormInput(
        **make_form(
            age=70, sex="M", day_ill=6,
            symptoms={**ALL_NO_SYMPTOMS, "FEBRE": "yes", "VOMITO": "yes",
                      "PETEQUIA_N": "yes", "LEUCOPENIA": "yes", "NAUSEA": "yes"},
            comorbidities={**ALL_NO_COMORB, "HEMATOLOG": "yes", "RENAL": "yes",
                           "DIABETES": "yes"},
        )
    )
    low = model.score_all(encode_features(healthy, ref))
    high = model.score_all(encode_features(sick, ref))

    assert high["A"].score > low["A"].score
    assert high["B"].score > low["B"].score
    assert high["B2"].score > low["B2"].score


def test_leucopenia_dominates_severity_model():
    """白细胞减少是重症模型最强的单一预测因子之一（B2 系数 1.4）。"""
    from app.ml_model import DengueModel, encode_features
    from app.schemas import FormInput

    ref = date(2026, 8, 16)
    model = DengueModel()
    base = make_form(
        age=40, sex="M", day_ill=3,
        symptoms=ALL_NO_SYMPTOMS, comorbidities=ALL_NO_COMORB,
    )
    without = model.score_one(
        "B2", encode_features(FormInput(**base), ref)
    )
    with_leuco = model.score_one(
        "B2",
        encode_features(
            FormInput(**{**base, "symptoms": {**ALL_NO_SYMPTOMS, "LEUCOPENIA": "yes"}}),
            ref,
        ),
    )
    assert with_leuco.z - without.z == pytest.approx(1.4, abs=1e-6)
    assert with_leuco.score > without.score


def test_who_warning_signs_flagged_independently_of_model(client):
    """WHO 警示征象是规则判断：即便三个模型评分都不高也必须给出。"""
    resp = client.post(
        "/api/assess",
        json=make_form(
            age=45, sex="M", day_ill=5,
            symptoms={**ALL_NO_SYMPTOMS, "FEBRE": "yes", "VOMITO": "yes",
                      "PETEQUIA_N": "yes"},
            comorbidities=ALL_NO_COMORB,
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["warning_signs"]) == {"VOMITO", "PETEQUIA_N"}
    # 这正是需要提示的场景：模型评分并不高
    assert body["severe"]["level"] in {"low", "medium"}


def test_no_warning_signs_when_not_reported(client):
    resp = client.post(
        "/api/assess",
        json=make_form(
            symptoms={**ALL_NO_SYMPTOMS, "FEBRE": "yes"},
            comorbidities=ALL_NO_COMORB,
        ),
    )
    assert resp.json()["warning_signs"] == []


def test_unknown_is_not_a_warning_sign(client):
    """「不知道」不能被当成报告了警示征象。"""
    resp = client.post(
        "/api/assess",
        json=make_form(
            symptoms={**ALL_NO_SYMPTOMS, "VOMITO": "unknown", "PETEQUIA_N": "unknown"},
            comorbidities=ALL_NO_COMORB,
        ),
    )
    assert resp.json()["warning_signs"] == []


def test_level_thresholds():
    from app.ml_model import _level

    assert _level(0.0) == "low"
    assert _level(34.9) == "low"
    assert _level(35.0) == "medium"
    assert _level(65.0) == "medium"
    assert _level(65.1) == "high"
    assert _level(100.0) == "high"


def test_epi_week_clamped_to_52():
    """ISO 第 53 周并入第 52 周，与训练时的 52 周编码一致。"""
    from app.ml_model import get_epi_week

    # 2026-12-31 属于 ISO 第 53 周
    assert date(2026, 12, 31).isocalendar().week == 53
    assert get_epi_week(date(2026, 12, 31)) == 52
