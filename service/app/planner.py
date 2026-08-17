"""自适应问诊规划器：决定下一个该问什么，并证明什么时候可以安全停止。

== 原理（完全确定性，无 LLM）==

三个逻辑回归模型的系数是完全已知的，因此对一份**部分作答**的问卷，
可以对每个模型算出最终分数的**硬边界**：

- 已作答的二值题贡献是确定的：yes -> coef，no / unknown -> 0
  （与 SINAN 特征工程一致，见 ml_model 模块文档）；
- 一道**还没问**的二值题，最终贡献只可能是 0 或它的系数，于是

      z_min = z_answered + Σ min(0, c_f)
      z_max = z_answered + Σ max(0, c_f)        （f 取遍未问特征）

- age / sex / day_ill 是问卷第一步的必填项，季节项由服务器按当日计算，
  两者都不带来不确定性；
- 把 z_min / z_max 送进与 score_one **完全相同**的参考人归一化
  （ml_model.score_from_z，裁剪到 [0, 100]），得到 [score_min, score_max]。

若三个模型的区间都落在同一个风险档位内（_level(score_min) == _level(score_max)），
则剩余问题**无论怎么回答**都改变不了任何模型的档位——可证明地安全停止。

关键区分：用户答了「不知道」是确定的 0（已作答）；「还没问」才是不确定。
PlanRequest 用键的存在与否区分这两种状态（见 schemas.PlanRequest 的说明）。

== 信息价值 ==

系数本身就定义了每道题的信息量。对未问特征 f：

    impact(f) = Σ_{未定模型 m} |c_f^m| / (z_ceil^m − z_ref^m)

除以各模型自己的归一化跨度，三个模型的系数才可以互相比较、相加。
why_model 取归一化 |系数| 最大的那个未定模型——前端用它解释
「这道题主要在帮哪条估计收窄」。排序完全确定：先按 impact 降序，
平手按 FEATS 顺序（症状在前、合并症在后，各自按训练脚本的次序）。
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

# 规划器管理的全部问题：(kind, code)，顺序 = FEATS 顺序（先症状后合并症）
QUESTIONS: tuple[tuple[str, str], ...] = tuple(
    [("symptom", code) for code in SYMPTOM_CODES]
    + [("comorbidity", code) for code in COMORB_CODES]
)
QUESTION_COUNT = len(QUESTIONS)  # 21

# next 列表最多返回几条建议
NEXT_MAX = 5


def _seasonal(ref_date: date | None) -> tuple[float, float]:
    """当日的季节项（与 encode_features 完全相同的公式）。"""
    week = get_epi_week(ref_date)
    return (
        math.sin(2 * math.pi * week / 52),
        math.cos(2 * math.pi * week / 52),
    )


def _answered_values(req: PlanRequest, wk_sin: float, wk_cos: float) -> dict[str, float]:
    """当前已知的 26 维特征值：yes -> 1，no / unknown / 未问 -> 0。

    「未问按 0 计」正是 score_now 的定义——用户此刻停止作答，
    缺失键会被 FormInput 补成 unknown，编码为 0，得到的就是这个分数。
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
    """还没问的二值特征名（键缺失 = 未问；已答 unknown 不在其列）。"""
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
    """单个模型的 [score_min, score_max] 硬边界与当前分。

    z 的求和顺序与 score_one 逐项一致（按 FEATS 遍历），
    保证 score_now 与 /api/assess 的分数在浮点意义上也完全相同。
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
    """未问问题按信息价值排序，取前 NEXT_MAX 条。

    impact(f) = Σ_{未定模型} |coef| / 归一化跨度；对所有未定模型都零影响的
    问题直接跳过——问它不可能改变任何还悬而未决的档位。
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
        for key in undecided:  # MODEL_KEYS 顺序；why_model 平手取靠前的模型
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

    # 确定性排序：impact 降序，平手按 FEATS 顺序
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [question for _, _, question in ranked[:NEXT_MAX]]


def plan(
    req: PlanRequest,
    ref_date: date | None = None,
    model: DengueModel | None = None,
) -> PlanResponse:
    """核心入口：边界 -> 是否可停 -> 下一步问题。纯函数，无副作用。

    ref_date / model 仅供测试注入；生产路径用当日日期与进程内单例模型。
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
        # 全部定档后不再有值得问的问题——即便还剩很多没问
        next=[] if can_stop else _rank_next(model, req, undecided, wk_sin, wk_cos),
        answered=answered,
        remaining=QUESTION_COUNT - answered,
    )
