"""三层数据契约：问卷输入 FormInput -> ML 特征 MLFeatures -> 评估结果 AssessmentResult。

特征定义严格对齐登革热风险模型的训练脚本（02_fit_models.py 的 FEATS），
顺序与命名不可更改，否则推理结果无意义。

三态答案（yes/no/unknown）的编码依据：训练数据 SINAN 用 1=有、2=无、9=未知，
特征工程里只有 "1" 记为 1，因此「无」与「不知道」在模型看来都是 0。
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------- 特征定义（与训练脚本一致，顺序不可改） ----------

# 14 个症状
SYMPTOM_CODES: tuple[str, ...] = (
    "FEBRE",       # 发热
    "MIALGIA",     # 肌痛
    "CEFALEIA",    # 头痛
    "EXANTEMA",    # 皮疹
    "VOMITO",      # 呕吐
    "NAUSEA",      # 恶心
    "DOR_COSTAS",  # 背痛
    "CONJUNTVIT",  # 结膜炎
    "ARTRITE",     # 关节炎
    "ARTRALGIA",   # 关节痛
    "PETEQUIA_N",  # 瘀点（皮肤出血点）
    "LEUCOPENIA",  # 白细胞减少（重症最强预测因子）
    "LACO",        # 束臂试验阳性
    "DOR_RETRO",   # 眼后痛
)

# 7 个合并症
COMORB_CODES: tuple[str, ...] = (
    "DIABETES",    # 糖尿病
    "HEMATOLOG",   # 血液病
    "HEPATOPAT",   # 肝病
    "RENAL",       # 肾病
    "HIPERTENSA",  # 高血压
    "ACIDO_PEPT",  # 消化性溃疡
    "AUTO_IMUNE",  # 自身免疫病
)

# 26 个特征的完整顺序（= 训练时 FEATS）
FEATS: tuple[str, ...] = (
    tuple(f"{c}_x" for c in SYMPTOM_CODES)
    + tuple(f"{c}_x" for c in COMORB_CODES)
    + ("age", "sex_f", "day_ill", "wk_sin", "wk_cos")
)

# ---------- 枚举类型 ----------

SymptomAnswer = Literal["yes", "no", "unknown"]
Sex = Literal["F", "M"]
RiskLevel = Literal["low", "medium", "high"]
Language = Literal["zh-CN", "zh-TW", "en", "es", "pt"]
ModelKey = Literal["A", "B", "B2"]

# ---------- 五语言固定文案 ----------

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

# 模型性质说明：相对评分而非概率（无截距 + 下采样训练，尚未本地校准）
MODEL_NOTES: dict[str, str] = {
    "zh-CN": (
        "评分为相对风险参考值，非感染概率。模型基于巴西 SINAN 2023–2025 年"
        "登革热通报数据训练，尚未在本地人群校准。"
    ),
    "zh-TW": (
        "評分為相對風險參考值，非感染機率。模型基於巴西 SINAN 2023–2025 年"
        "登革熱通報資料訓練，尚未在本地人群校準。"
    ),
    "en": (
        "Scores are relative risk indicators, not infection probabilities. The model was "
        "trained on Brazilian SINAN dengue surveillance data (2023–2025) and has not been "
        "calibrated for local populations."
    ),
    "es": (
        "Las puntuaciones son indicadores de riesgo relativo, no probabilidades de infección. "
        "El modelo se entrenó con datos de vigilancia de dengue de SINAN Brasil (2023–2025) "
        "y no está calibrado para poblaciones locales."
    ),
    "pt": (
        "As pontuações são indicadores de risco relativo, não probabilidades de infecção. "
        "O modelo foi treinado com dados de vigilância de dengue do SINAN Brasil (2023–2025) "
        "e não foi calibrado para populações locais."
    ),
}

# 默认（简体中文）文案
DISCLAIMER = DISCLAIMERS["zh-CN"]

# WHO 登革热警示征象（Guidelines for Diagnosis, Treatment, Prevention and Control, 2009）
# 中能被本问卷覆盖的项。这是**独立于模型的规则判断**：
# 模型 B/B2 由白细胞减少主导，未验血的患者即便已出现警示征象也可能得低分，
# 因此必须并列给出规则提示，避免用户被错误安抚。
WARNING_SIGN_CODES: tuple[str, ...] = (
    "VOMITO",      # 持续呕吐
    "PETEQUIA_N",  # 皮肤黏膜出血表现
)


class FormInput(BaseModel):
    """POST /api/assess 请求体：登革热风险自评问卷。"""

    age: int = Field(..., ge=0, le=110, description="年龄（岁）")
    sex: Sex = Field(..., description="生理性别，F 女 / M 男")
    day_ill: int = Field(..., ge=0, le=14, description="症状开始至今的天数")
    symptoms: dict[str, SymptomAnswer] = Field(
        default_factory=dict, description="14 项症状，缺失的键按 unknown 处理"
    )
    comorbidities: dict[str, SymptomAnswer] = Field(
        default_factory=dict, description="7 项合并症，缺失的键按 unknown 处理"
    )
    language: Language = Field(default="zh-CN", description="输出语言（BCP 47）")
    notes: str = Field(default="", max_length=500, description="可选自由文本补充说明")

    @field_validator("symptoms")
    @classmethod
    def _fill_symptoms(cls, value: dict[str, str]) -> dict[str, str]:
        return _fill_answers(value, SYMPTOM_CODES, "symptoms")

    @field_validator("comorbidities")
    @classmethod
    def _fill_comorbidities(cls, value: dict[str, str]) -> dict[str, str]:
        return _fill_answers(value, COMORB_CODES, "comorbidities")


def _fill_answers(value: dict, codes: tuple[str, ...], field: str) -> dict:
    """补全缺失的键为 unknown；出现契约外的键则报错。"""
    unknown_keys = set(value) - set(codes)
    if unknown_keys:
        raise ValueError(f"{field} 含未知字段：{sorted(unknown_keys)}")
    return {code: value.get(code, "unknown") for code in codes}


class MLFeatures(BaseModel):
    """26 个模型输入特征。二值项 0/1，其余为连续值。"""

    # 14 个症状
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
    # 7 个合并症
    DIABETES_x: int = Field(..., ge=0, le=1)
    HEMATOLOG_x: int = Field(..., ge=0, le=1)
    HEPATOPAT_x: int = Field(..., ge=0, le=1)
    RENAL_x: int = Field(..., ge=0, le=1)
    HIPERTENSA_x: int = Field(..., ge=0, le=1)
    ACIDO_PEPT_x: int = Field(..., ge=0, le=1)
    AUTO_IMUNE_x: int = Field(..., ge=0, le=1)
    # 连续 / 其他
    age: float = Field(..., ge=0.0, le=110.0)
    sex_f: float = Field(..., ge=0.0, le=1.0)
    day_ill: float = Field(..., ge=0.0, le=14.0)
    wk_sin: float = Field(..., ge=-1.0, le=1.0)
    wk_cos: float = Field(..., ge=-1.0, le=1.0)

    def as_vector(self) -> list[float]:
        """按 FEATS 顺序展开为特征向量（供外部模型使用）。"""
        data = self.model_dump()
        return [float(data[name]) for name in FEATS]


class ModelScore(BaseModel):
    """单个模型的输出：相对风险评分。"""

    score: float = Field(..., ge=0.0, le=100.0, description="0-100 相对评分")
    level: RiskLevel
    z: float = Field(..., description="线性预测值（无截距）")


class Advice(BaseModel):
    """三类建议，均为目标语言（FormInput.language）字符串列表。"""

    protection: list[str]
    medical: list[str]
    monitoring: list[str]


class AssessmentResult(BaseModel):
    """POST /api/assess 响应体。"""

    dengue: ModelScore      # 模型 A：是否登革热
    worsening: ModelScore   # 模型 B：是否加重（警示+重症 vs 普通）
    severe: ModelScore      # 模型 B2：是否重症
    epi_week: int = Field(..., ge=1, le=52, description="评估当日的流行病学周")
    warning_signs: list[str] = Field(
        default_factory=list,
        description="用户报告的 WHO 登革热警示征象代码（规则判断，独立于模型评分）",
    )
    summary: str
    advice: Advice
    disclaimer: str = DISCLAIMER
    model_note: str = MODEL_NOTES["zh-CN"]
