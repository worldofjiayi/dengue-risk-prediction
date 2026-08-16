"""三层数据契约：问卷输入 FormInput -> ML 特征 MLFeatures -> 评估结果 AssessmentResult。

特征定义严格对齐登革热风险模型的训练脚本（02_fit_models.py 的 FEATS），
顺序与命名不可更改，否则推理结果无意义。

三态答案（yes/no/unknown）的编码依据：训练数据 SINAN 用 1=有、2=无、9=未知，
特征工程里只有 "1" 记为 1，因此「无」与「不知道」在模型看来都是 0。

⚠️ 流行病学暴露问题（EXPOSURE_CODES）**不是模型特征**：SINAN 通报数据里没有
这三个变量，硬塞进 26 维向量会破坏与训练脚本的一致性。它们只走
「规则判断」通道，产出独立的 ExposureContext，详见该类的说明。
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

# 5 个非二值特征（在解释输出里保留原名，不做 _x 剥离）
NON_BINARY_FEATS: tuple[str, ...] = ("age", "sex_f", "day_ill", "wk_sin", "wk_cos")

# 26 个特征的完整顺序（= 训练时 FEATS）
FEATS: tuple[str, ...] = (
    tuple(f"{c}_x" for c in SYMPTOM_CODES)
    + tuple(f"{c}_x" for c in COMORB_CODES)
    + NON_BINARY_FEATS
)

# ---------- 流行病学暴露（规则通道，**不是**模型特征） ----------
#
# 这三个问题在 SINAN 通报数据里根本不存在，因此无法进入逻辑回归模型：
# 训练时没见过的变量没有系数，强行加入只会让 26 维向量与训练脚本不再对齐。
# 但「身边有确诊病例」「去过暴发地区」是登革热问诊中最重要的线索之一，
# 丢掉它可惜。折中方案是把它们放进一条**独立的规则**（见 pipeline.evaluate_exposure），
# 与模型评分并列呈现、互不干扰——和 WHO 警示征象采用的是同一种设计。
EXPOSURE_CODES: tuple[str, ...] = (
    "FEVER_CLUSTER",    # 周围人群近期发热病例异常增多
    "CONFIRMED_CASE",   # 身边（家庭 / 工作场所 / 社区）有确诊登革热病例
    "OUTBREAK_TRAVEL",  # 近期到访或居住于登革热暴发地区
)

# 判定为 high 的暴露因素（任一为 yes 即 high）
HIGH_EXPOSURE_CODES: tuple[str, ...] = ("CONFIRMED_CASE", "OUTBREAK_TRAVEL")
# 判定为 medium 的暴露因素（仅在不满足 high 时生效）
MEDIUM_EXPOSURE_CODES: tuple[str, ...] = ("FEVER_CLUSTER",)

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

# 上游模型不可用时返回给用户的提示（HTTP 502）
UPSTREAM_ERRORS: dict[str, str] = {
    "zh-CN": "上游模型服务暂时不可用，请稍后重试。",
    "zh-TW": "上游模型服務暫時無法使用，請稍後重試。",
    "en": "The upstream model service is temporarily unavailable. Please try again shortly.",
    "es": "El servicio del modelo no está disponible temporalmente. Inténtelo de nuevo más tarde.",
    "pt": "O serviço do modelo está temporariamente indisponível. Tente novamente em instantes.",
}

# 服务端内部错误提示（HTTP 500）
SERVER_ERRORS: dict[str, str] = {
    "zh-CN": "服务器内部错误，请稍后重试。",
    "zh-TW": "伺服器內部錯誤，請稍後重試。",
    "en": "Internal server error. Please try again later.",
    "es": "Error interno del servidor. Inténtelo de nuevo más tarde.",
    "pt": "Erro interno do servidor. Tente novamente mais tarde.",
}

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
    # validate_default=True：整个字段缺席时也要跑补全校验器，
    # 否则 form.symptoms 会是空 dict，下游按代码取值就会 KeyError。
    symptoms: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        validate_default=True,
        description="14 项症状，缺失的键按 unknown 处理",
    )
    comorbidities: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        validate_default=True,
        description="7 项合并症，缺失的键按 unknown 处理",
    )
    exposure: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        validate_default=True,
        description=(
            "3 项流行病学暴露问题，缺失的键按 unknown 处理。"
            "**不参与模型打分**，只用于规则化的 exposure_context。"
        ),
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

    @field_validator("exposure")
    @classmethod
    def _fill_exposure(cls, value: dict[str, str]) -> dict[str, str]:
        return _fill_answers(value, EXPOSURE_CODES, "exposure")


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


class ExposureContext(BaseModel):
    """流行病学暴露背景：**规则判断结果，不来自任何模型**。

    level：
        high   —— CONFIRMED_CASE 或 OUTBREAK_TRAVEL 为 yes
        medium —— FEVER_CLUSTER 为 yes 且未达 high
        low    —— 其余情况（含全部「不知道」）
    factors：回答为 yes 的暴露代码，供前端查表显示本地化标签。

    之所以不并入模型评分：这三个变量在 SINAN 训练数据中不存在，
    没有系数可用，任何加权都会是拍脑袋的数字。分开呈现才诚实。
    """

    level: RiskLevel
    factors: list[str] = Field(default_factory=list)


class FeatureContribution(BaseModel):
    """单个特征对某模型线性预测值 z 的贡献（coef × 特征值）。"""

    feature: str = Field(..., description="FEATS 中的特征名，如 FEBRE_x")
    code: str = Field(
        ...,
        description="供前端查表的标签键：症状/合并症去掉 _x 后缀，5 个非二值特征用原名",
    )
    contribution: float = Field(..., description="coef × 特征值，保留 4 位小数")
    direction: Literal["up", "down"] = Field(..., description="推高（up）还是拉低（down）")


class Advice(BaseModel):
    """三类建议，均为目标语言（FormInput.language）字符串列表。

    字段顺序即前端展示顺序：**就医优先**，其次居家监测，最后日常防护。
    Pydantic 按字段声明顺序序列化，改动此处会直接改变响应 JSON 的键顺序。
    """

    medical: list[str]
    monitoring: list[str]
    protection: list[str]


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
    exposure_context: ExposureContext = Field(
        ...,
        description="流行病学暴露背景（规则判断，独立于模型评分）",
    )
    summary: str
    advice: Advice
    explanations: dict[str, list[FeatureContribution]] = Field(
        default_factory=dict,
        description="每个模型 z 的前 5 大贡献项，键为 dengue / worsening / severe",
    )
    disclaimer: str = DISCLAIMER
    model_note: str = MODEL_NOTES["zh-CN"]


# ---------- 追问对话（POST /api/chat） ----------

# 历史消息上限：只保留最近 N 条，超出部分静默截断（比 422 更友好）
CHAT_HISTORY_MAX = 6
CHAT_QUESTION_MAX = 500


class ChatScore(BaseModel):
    """前端回传的单模型评分（AssessmentResult.ModelScore 的精简版）。"""

    score: float = Field(..., ge=0.0, le=100.0)
    level: RiskLevel


class ChatContext(BaseModel):
    """用户自己那份评估结果的快照。服务端无状态，全部由前端回传。"""

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
    """一轮历史消息。"""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    """POST /api/chat 请求体。"""

    language: Language = "zh-CN"
    question: str = Field(..., min_length=1, max_length=CHAT_QUESTION_MAX)
    context: ChatContext = Field(default_factory=ChatContext)
    history: list[ChatMessage] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question 不能为空白")
        return text

    @field_validator("history")
    @classmethod
    def _truncate_history(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        # 截断而非报错：用户不该因为聊得久而被 422 拦下
        return value[-CHAT_HISTORY_MAX:]


class ChatResponse(BaseModel):
    """POST /api/chat 响应体。"""

    reply: str


def _drop_unknown_keys(value: dict, codes: tuple[str, ...]) -> dict:
    """只保留契约内的键。

    与 FormInput 的严格校验不同：/api/chat 的上下文是前端回传的快照，
    多一个陌生键不该让用户问不了问题，静默丢弃即可。
    """
    return {k: v for k, v in value.items() if k in codes}
