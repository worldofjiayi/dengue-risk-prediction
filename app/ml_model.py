"""登革热风险模型推理。

== 模型来源 ==
巴西 SINAN 法定传染病通报系统 2023–2025 年 944.99 万条登革热通报数据训练的
三个逻辑回归模型，系数存放在 app/model/dengue_models.json：

  A  —— 是否登革热（确诊 vs 不确定）        AUC 0.686
  B  —— 是否加重（警示+重症 vs 普通）        AUC 0.722
  B2 —— 是否重症（重症 vs 其他登革热）       AUC 0.810（最强）

== 三个必须知道的限制 ==

1. **无截距**。训练脚本只导出了 coef_，没有导出 intercept_，因此 z 是不含常数项的
   线性预测值。经 sigmoid 得到的 0-100 分只能作为**相对风险排序**使用。

2. **训练做了下采样 + class_weight="balanced"**。即便有截距，其对应的也是重采样后的
   人为患病率，而非真实人群患病率。所以本模块的输出**不是感染概率**，
   任何面向用户的文案都不得把它表述为百分比概率。

3. **阈值未校准**。low/medium/high 的分档是工程上的默认切分，
   部署到真实人群前需要在保持原始患病率的测试集上重新评估
   （项目 README「已知局限」一节的原话）。

== 0-100 分怎么来的 ==

不能用 sigmoid(z)：没有截距时 z 恒偏正，绝大多数有症状的人都会逼近 100 分。
也不能用「理论最小值」做下界：z_min 对应的是「拥有全部负系数症状」这种反常状态，
真正无症状的人反而会落在区间中部（实测健康年轻人被判成 medium）。

采用**参考人锚定**：

    score = 100 × (z − z_ref) / (z_ceil − z_ref)      结果裁剪到 [0, 100]

    z_ref  = 同一季节、无任何症状与合并症、30 岁男性、病程 0 天
    z_ceil = 同一季节、所有升高风险的特征都取到上界（age 110、day_ill 14、二值取 1）

分数的含义因此是：**相对于此刻一个无症状的人，你在这个模型上处在多高的位置**。
季节项在 z、z_ref、z_ceil 中相同，因而在比值里自然抵消——这是刻意的：
wk_sin/wk_cos 描述的是人群层面的季节基线，不是个体风险差异。
（季节项仍参与 z 的计算并写入评测回流，供将来做本地校准时使用。）

这依然只是**相对风险指数**，与概率无关。

== 特征编码 ==
训练数据 SINAN 用 1=有、2=无、9=未知，特征工程里 (df[c] == "1") 意味着
「无」和「未知」都编码为 0。本模块的 yes→1 / no→0 / unknown→0 与之一致。

季节项 wk_sin / wk_cos 由服务器按评估当日的 ISO 周计算，用户不填写。
⚠️ 模型训练于南半球（巴西）数据，季节项对北半球用户方向相反；
赤道地区季节性本身较弱，影响有限，但这是已知的迁移局限。
"""

import json
import logging
import math
from datetime import date
from pathlib import Path

from app.schemas import (
    COMORB_CODES,
    FEATS,
    SYMPTOM_CODES,
    FormInput,
    MLFeatures,
    ModelScore,
)

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).resolve().parent / "model" / "dengue_models.json"

# 风险分档阈值（工程默认值，未经人群校准）
_LOW_MAX = 35.0
_MEDIUM_MAX = 65.0

# 模型键 -> 结果字段名
MODEL_KEYS: tuple[str, ...] = ("A", "B", "B2")


# 参考人的年龄（用于锚定 0 分基准，不随用户年龄变化）
_REF_AGE = 30.0
# 特征上界（与 FormInput 的校验范围一致）
_AGE_MAX = 110.0
_DAY_ILL_MAX = 14.0
# 季节项特征名（在参考人与上界中原样保留，使其在归一化时抵消）
_SEASON_FEATS = ("wk_sin", "wk_cos")


def _season_part(coef: dict[str, float], wk_sin: float, wk_cos: float) -> float:
    return coef.get("wk_sin", 0.0) * wk_sin + coef.get("wk_cos", 0.0) * wk_cos


def _reference_z(coef: dict[str, float], wk_sin: float, wk_cos: float) -> float:
    """参考人：同季节、无任何症状与合并症、30 岁男性、病程 0 天。"""
    return coef.get("age", 0.0) * _REF_AGE + _season_part(coef, wk_sin, wk_cos)


def _ceiling_z(coef: dict[str, float], wk_sin: float, wk_cos: float) -> float:
    """上界：同季节，所有会升高风险的特征都取到上界。"""
    z = _season_part(coef, wk_sin, wk_cos)
    for name in FEATS:
        if name in _SEASON_FEATS:
            continue
        c = coef.get(name, 0.0)
        if name == "age":
            z += max(0.0, c * _AGE_MAX)
        elif name == "day_ill":
            z += max(0.0, c * _DAY_ILL_MAX)
        else:  # 二值特征（含 sex_f）：取 0 与 1 中贡献更大的一侧
            z += max(0.0, c)
    return z


def _load_models() -> dict[str, dict]:
    """读取模型系数文件；缺列会在打分时按 0 处理并告警。"""
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
    """评估当日的流行病学周（1-52）。第 53 周并入第 52 周，与训练时的 52 周编码一致。"""
    ref = ref_date or date.today()
    return min(ref.isocalendar().week, 52)


def _answer_to_int(answer: str) -> int:
    """yes -> 1；no / unknown -> 0（与 SINAN 特征工程一致）。"""
    return 1 if answer == "yes" else 0


def encode_features(form: FormInput, ref_date: date | None = None) -> MLFeatures:
    """问卷答案 -> 26 个模型特征（确定性编码，不依赖任何外部服务）。"""
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


def _level(score: float) -> str:
    if score < _LOW_MAX:
        return "low"
    if score <= _MEDIUM_MAX:
        return "medium"
    return "high"


class DengueModel:
    """三个逻辑回归模型的推理封装。"""

    def __init__(self, models: dict[str, dict] | None = None) -> None:
        self._models = models if models is not None else _MODELS

    def score_one(self, key: str, features: MLFeatures) -> ModelScore:
        """单个模型打分。

        z = Σ coef × feature（无截距）；
        score = z 相对「同季节无症状参考人」的位置，上界为全风险因子拉满，
        结果裁剪到 [0, 100]。详见模块文档。
        """
        coef = self._models[key]["coef"]
        data = features.model_dump()
        z = sum(coef.get(name, 0.0) * float(data[name]) for name in FEATS)

        z_ref = _reference_z(coef, data["wk_sin"], data["wk_cos"])
        z_ceil = _ceiling_z(coef, data["wk_sin"], data["wk_cos"])
        span = z_ceil - z_ref
        ratio = (z - z_ref) / span if span > 0 else 0.0
        score = round(100.0 * max(0.0, min(1.0, ratio)), 1)
        return ModelScore(score=score, level=_level(score), z=round(z, 4))

    def score_all(self, features: MLFeatures) -> dict[str, ModelScore]:
        """三个模型全部打分，返回 {"A":…, "B":…, "B2":…}。"""
        return {key: self.score_one(key, features) for key in MODEL_KEYS}

    def info(self) -> dict[str, dict]:
        """模型元信息（名称与 AUC），供 /api/health 等展示。"""
        return {
            key: {"name": m["name"], "auc": m["auc"]}
            for key, m in self._models.items()
        }


_model: DengueModel | None = None


def get_model() -> DengueModel:
    """进程内单例。"""
    global _model
    if _model is None:
        _model = DengueModel()
        logger.info("登革热模型已加载：%s", list(_model.info()))
    return _model
