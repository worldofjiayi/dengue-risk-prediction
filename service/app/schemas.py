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
# 建议文本的来源：真实模型输出（已通过输出校验）还是内置模板兜底
AdviceSource = Literal["llm", "template"]
# 一条引用来源出自哪一层：WHO 疾病暴发新闻接口，还是模型的联网检索
SourceOrigin = Literal["who", "search"]
SourceAuthority = Literal["official", "other"]
# 联网检索这一轮的状态：
#   ok       —— 检索跑了，并且带回了来源
#   degraded —— 检索被尝试过，但失败 / 一无所获 / 输出没通过校验（其余各层照常返回）
#   disabled —— SEARCH_ENABLED=false，压根没打算检索
SearchStatus = Literal["ok", "degraded", "disabled"]
# 地区流行程度（与 app/data/dengue_endemicity.json 的取值一致）
Endemicity = Literal["high", "moderate", "low", "none", "unknown"]

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
    _require_known_keys(value, codes, field)
    return {code: value.get(code, "unknown") for code in codes}


def _require_known_keys(value: dict, codes: tuple[str, ...], field: str) -> dict:
    """只校验键在契约内，**不补全缺失键**。

    /api/plan 的语义是「键存在 = 已作答，键缺失 = 还没问」，
    缺失本身就是信号，绝不能像 FormInput 那样补成 unknown。
    """
    unknown_keys = set(value) - set(codes)
    if unknown_keys:
        raise ValueError(f"{field} 含未知字段：{sorted(unknown_keys)}")
    return value


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
    advice_source: AdviceSource = Field(
        default="template",
        description=(
            "这段建议是谁写的：llm = 真实模型生成且通过了输出校验；"
            "template = 内置模板（演示模式，或模型失败/两次都没通过校验后的兜底）。"
            "评分与规则判断在两种情况下都是真实计算的，只有自然语言部分不同。"
        ),
    )


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


class Source(BaseModel):
    """一条可引用的来源。

    origin 说明这条链接是**哪一层**给的：
        who    —— app.intel 从 WHO 疾病暴发新闻接口取回的通报（稳定、免费、总是可用）
        search —— 模型这一轮联网检索真正返回的页面（按次计费，可能一无所获）
    两层的可信度与时效性不同，前端要能分别标注，所以标签必须跟着链接一起走，
    而不是靠 URL 前缀去猜。

    authority 是**同一层内部**的第二个区分：检索结果里，一国卫生部的通报和
    一个新闻聚合站会并排出现（实测新加坡那次，nea.gov.sg 与 magzter.com 同列）。
    两者可核对程度差得远，所以按域名判定是否政府/国际卫生机构，让前端能标出来。
    判定只看域名，不看内容——这是个可核对的事实，不是对质量的评价。

    date 允许为空：WHO 通报一定有发布日期，检索结果经常没有。
    """

    title: str = Field(..., description="页面标题，原样取自接口/检索结果")
    date: str | None = Field(default=None, description="发布日期，取不到时为 null")
    url: str = Field(..., description="来源页面地址")
    origin: SourceOrigin = Field(default="who", description="这条来源来自哪一层")
    authority: SourceAuthority = Field(
        default="other", description="域名是否属于政府或国际卫生机构"
    )


class ChatResponse(BaseModel):
    """POST /api/chat 响应体。

    sources 是**这一轮真正取回过的**来源：WHO 工具结果 + 联网检索结果的并集。
    它同时是给用户看的引用列表，和校验器判断「回复里的链接是不是编的」所依据的
    白名单——回复中出现任何不在这个列表里的链接，这一轮就会被判失败并退回兜底文案。
    没查任何东西（或什么都没查到）时它就是空列表，回复里也不该有任何链接。
    """

    reply: str
    sources: list[Source] = Field(default_factory=list)
    search_count: int = Field(
        default=0,
        ge=0,
        description=(
            "这一轮真的联网检索了几次。没有识别到地点、或 SEARCH_ENABLED=false 时恒为 0——"
            "检索按次计费，花没花钱不该只有服务端日志知道。"
        ),
    )


# ---------- 目的地查询（POST /api/destination） ----------

DESTINATION_LOCATION_MAX = 120


class WhoNotice(BaseModel):
    """一条 WHO 疾病暴发新闻通报（形状与 intel.lookup_dengue_context 一致）。"""

    title: str
    date: str
    url: str


class DestinationAdvice(BaseModel):
    """出行前的三类建议。

    字段顺序**刻意与 Advice 不同**：这里没有病人、也没有评分，用户是在出发前问
    「那边现在什么情况」。最该先说的是怎么不被叮，其次才是「什么情况下去看医生」
    与旅途中该盯着什么。Pydantic 按声明顺序序列化，这个顺序就是前端展示顺序。
    """

    protection: list[str]
    medical: list[str]
    monitoring: list[str]


class DestinationRequest(BaseModel):
    """POST /api/destination 请求体。"""

    location: str = Field(
        ...,
        min_length=1,
        max_length=DESTINATION_LOCATION_MAX,
        description="国家 / 地区 / 城市名，任意语言",
    )
    language: Language = Field(default="zh-CN", description="输出语言（BCP 47）")

    @field_validator("location")
    @classmethod
    def _location_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("location 不能为空白")
        return text


class DestinationResponse(BaseModel):
    """POST /api/destination 响应体。

    **这里没有任何评分**，一个也没有。地点从来不参与打分，也不改变暴露档位；
    它是行前参考，与三个模型输出属于两件事。硬要在这里返回一个「目的地风险分」，
    就是凭一张粗粒度国家表编出一个数字。

    三层结构，可信度从高到低：
      1. endemicity / season_note / who_notices —— 本地表 + WHO 接口，稳定且免费；
      2. recent_findings —— 模型联网检索得到的「最近三个月」要点，按次计费，可能没有；
      3. advice —— 与档位对应的固定文案，永远可用。
    第 2 层没拿到时降级到 1+3，绝不用「常识」把它填满。
    """

    location: str = Field(..., description="规范英文名；没认出来就是原样输入")
    matched: bool = Field(..., description="是否在内置地区表里认出了这个地名")
    endemicity: Endemicity
    season_note: str | None = Field(default=None, description="季节/地域说明；未命中为 null")
    who_notices: list[WhoNotice] = Field(default_factory=list)
    recent_findings: list[str] = Field(
        default_factory=list,
        description="最近约三个月的事实要点；检索失败或未通过校验时为空列表",
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="WHO 通报（origin=who）与检索结果（origin=search）的合并去重清单",
    )
    advice: DestinationAdvice
    search_status: SearchStatus
    disclaimer: str = DISCLAIMER
    model_note: str = MODEL_NOTES["zh-CN"]


# 一次请求最多向用户展示几条检索来源。
# 实测：两轮检索带回了 2 × 10 = 20 条结果，其中大半是新闻聚合站与不相关页面。
# 20 条链接不是引用列表，是噪音。
MAX_SEARCH_SOURCES = 8


def select_search_sources(
    sources: list[dict] | None, reply: str, limit: int = MAX_SEARCH_SOURCES
) -> list[dict]:
    """从检索返回的一大堆结果里挑出要展示（也就是要进白名单）的那些。

    **回复里真的引用过的链接一条都不能丢**：sources 同时是引用列表和校验器的
    白名单，把模型真正引用的那条截掉，就会被自己的校验器判成「编造链接」，
    好答案反而被退回兜底文案。所以先收全被引用的，再按原顺序补足到 limit。
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


# 政府域名标记：出现在国家顶级域之前（nea.gov.sg / doh.gov.ph / moph.go.th /
# gob.mx / gouv.fr / govt.nz）。单独一个 "go" 很危险（go.com 不是政府），
# 所以只在它后面跟着两字母国家码时才算数。
_GOV_LABELS = frozenset({"gov", "gob", "go", "gouv", "govt"})

# 没有政府域名但确属国际卫生/公共机构的，单独列出。宁可漏判也不错判：
# 判成 official 是在告诉用户「这条更可核对」，错标的代价比漏标大。
_OFFICIAL_HOSTS = frozenset(
    {
        "who.int",
        "paho.org",
        "europa.eu",       # ECDC 挂在 ecdc.europa.eu
        "un.org",
        "unicef.org",
    }
)


def classify_authority(url: str) -> SourceAuthority:
    """按域名判断这条来源是不是政府 / 国际卫生机构。

    只看域名，不看内容——这样结论是可核对的事实，而不是对报道质量的评价。
    一个卫生部的疫情通报和一个新闻聚合站的转载，读者有权一眼分清。

    判定规则（任一命中即 official）：
      · 顶级域是 gov 或 int          —— cdc.gov / who.int
      · 倒数第二段是政府标记且顶级域是两字母国家码 —— nea.gov.sg / moph.go.th / gob.mx
      · 域名或其父域在 _OFFICIAL_HOSTS 里 —— ecdc.europa.eu
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
    """把两层来源合成一个带 origin 标签的列表：WHO 通报在前，检索结果在后。

    放在 schemas 里而不是某个流水线模块里，是因为 /api/chat 与 /api/destination
    都要用它，而「一条来源长什么样、来自哪一层」本来就是数据契约的一部分。

    按 url 去重、保留顺序；WHO 在前是因为它更稳定也更容易核对。
    date 取不到就是 None（检索结果经常没有 page_age），绝不用今天的日期顶上。

    检索那一段内部再把 official 的排到前面：模型引用与否是它自己的事，但
    「哪几条来自卫生部门」应该先映入读者眼帘。**只调顺序，一条都不丢**——
    sources 同时是校验器的白名单，删掉任何一条都可能让正确回复被判成编造链接。
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
                authority="official",  # WHO 通报按定义就是官方来源
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
    # 稳定排序：official 在前，其余保持检索返回的原顺序
    found.sort(key=lambda s: 0 if s.authority == "official" else 1)
    merged.extend(found)
    return merged


def _drop_unknown_keys(value: dict, codes: tuple[str, ...]) -> dict:
    """只保留契约内的键。

    与 FormInput 的严格校验不同：/api/chat 的上下文是前端回传的快照，
    多一个陌生键不该让用户问不了问题，静默丢弃即可。
    """
    return {k: v for k, v in value.items() if k in codes}


# ---------- 自适应问诊规划（POST /api/plan） ----------


class PlanRequest(BaseModel):
    """POST /api/plan 请求体：一份**部分作答**的问卷。

    与 FormInput 的关键差异：**键的存在与否本身携带信息**。
      - 键存在 = 该问题已问过（yes / no / unknown 都是确定的回答）；
      - 键缺失 = 该问题还没问，最终取值不确定。

    因此不能复用 FormInput——它的校验器会把缺失键补成 unknown，
    恰好抹掉「已答不知道」与「还没问」的区别，而这正是规划器的全部依据。
    age / sex / day_ill 是问卷第一步的必填项，规划从它们已知开始。
    """

    age: int = Field(..., ge=0, le=110, description="年龄（岁）")
    sex: Sex = Field(..., description="生理性别，F 女 / M 男")
    day_ill: int = Field(..., ge=0, le=14, description="症状开始至今的天数")
    symptoms: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        description="已作答的症状：键存在 = 已问。缺失键 = 未问，不做补全。",
    )
    comorbidities: dict[str, SymptomAnswer] = Field(
        default_factory=dict,
        description="已作答的合并症，语义同 symptoms。",
    )
    language: Language = Field(default="zh-CN", description="输出语言（用于错误信息本地化）")

    @field_validator("symptoms")
    @classmethod
    def _known_symptom_keys(cls, value: dict[str, str]) -> dict[str, str]:
        return _require_known_keys(value, SYMPTOM_CODES, "symptoms")

    @field_validator("comorbidities")
    @classmethod
    def _known_comorb_keys(cls, value: dict[str, str]) -> dict[str, str]:
        return _require_known_keys(value, COMORB_CODES, "comorbidities")


class ModelBounds(BaseModel):
    """单个模型在部分作答下的分数硬边界（同一归一化，同一分档）。"""

    score_now: float = Field(
        ..., ge=0.0, le=100.0,
        description="未问按 0 计的当前分——用户此刻停止作答时的最终分",
    )
    score_min: float = Field(
        ..., ge=0.0, le=100.0, description="剩余问题任意作答都到不了更低的分数下界"
    )
    score_max: float = Field(
        ..., ge=0.0, le=100.0, description="剩余问题任意作答都到不了更高的分数上界"
    )
    level_now: RiskLevel
    decided: bool = Field(
        ..., description="[score_min, score_max] 是否已落在同一风险档位内"
    )


class PlanBounds(BaseModel):
    """三个模型的分数边界，键名与 AssessmentResult 一致。"""

    dengue: ModelBounds
    worsening: ModelBounds
    severe: ModelBounds


class NextQuestion(BaseModel):
    """建议接下来问的一道题。"""

    kind: Literal["symptom", "comorbidity"]
    code: str = Field(..., description="SYMPTOM_CODES / COMORB_CODES 中的问题代码")
    why_model: Literal["dengue", "worsening", "severe"] = Field(
        ..., description="这道题主要在收窄哪个尚未定档模型的估计"
    )


class PlanResponse(BaseModel):
    """POST /api/plan 响应体。"""

    bounds: PlanBounds
    can_stop: bool = Field(
        ..., description="三个模型全部 decided：任何剩余答案都无法改变任何档位"
    )
    next: list[NextQuestion] = Field(
        default_factory=list,
        description="最多 5 条，按信息价值降序；全部定档后恒为空",
    )
    answered: int = Field(..., ge=0, description="已作答的症状 + 合并症数量")
    remaining: int = Field(..., ge=0, description="尚未问的症状 + 合并症数量")
