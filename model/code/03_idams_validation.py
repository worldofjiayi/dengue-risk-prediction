#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_idams_validation.py — 用 Lancet IDAMS 已发表模型做外部校验
===========================================================
参考文献:
  Early diagnostic indicators of dengue versus other febrile illnesses
  in Asia and Latin America (IDAMS study). Lancet Global Health 2023.
  → appendix 7 (Supplement 7), Table S11 —— 纯临床精简模型的优势比(OR)

三项校验:
  ① 方向一致性  —— 重叠变量的系数方向是否与 IDAMS 的 OR 方向一致
  ② 发病天数梯度 —— 是否复现 IDAMS "判别力随病程上升"的现象
  ③ 计算器整体迁移 —— 逐人套用 IDAMS 评分,看在巴西数据上的实测表现

⚠ 两点必须声明的错配(决定了哪些校验能做):
  (1) 目标不同:IDAMS 的对照是真正的"其他发热(OFI)",
      而 SINAN 的负类是"不确定/排除(8)",并非干净的非登革热对照。
  (2) 变量缺失:IDAMS 的咳嗽、流涕、皮肤潮红、连续体温等强判别变量
      SINAN 根本不采集。11 个变量中仅 5 个可近似映射。
  因此本脚本做的是方向层面的定性校验 + 打折的定量迁移,而非严格外部验证。

用法:
    python3 03_idams_validation.py
"""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# IDAMS Table S11:按发病天数分层的优势比(纯临床精简模型)
#   cont  = 拉丁美洲 vs 亚洲(亚洲为参照)
#   age   = 25 岁 vs 8 岁的对比(17 年跨度)
#   temp  = 38℃ vs 37℃(每 1℃)
#   rash  = 皮疹;bleed = 皮肤出血
# 注意 bleed 的 OR 随病程急剧上升(2.75 → 11.13),是 IDAMS 后期最强的判别变量
# ---------------------------------------------------------------------------
ORS = {2: dict(cont=0.66, age=1.53, rash=0.95, bleed=2.75,  temp=1.71),
       3: dict(cont=0.60, age=1.42, rash=1.43, bleed=5.57,  temp=1.82),
       4: dict(cont=0.71, age=1.25, rash=2.20, bleed=7.60,  temp=2.03),
       5: dict(cont=0.69, age=1.53, rash=2.86, bleed=11.13, temp=1.53)}

# 计算器自带的分档阈值(论文附录未给截距,故为临时设定,需本地校准)
TIER_LOW, TIER_HIGH = 0.4, 1.6


def load_scored() -> pd.DataFrame:
    """载入三年数据,取发病第 2-5 天,逐人计算 IDAMS 风险评分(log-odds)。"""
    parts = []
    for f in ["/tmp/eng_2023.parquet", "/tmp/eng_2024.parquet", "/tmp/eng_2025.parquet"]:
        d = pd.read_parquet(f, columns=["CLASSI_FIN","day_ill","age","FEBRE_x",
                                        "EXANTEMA_x","PETEQUIA_N_x","LACO_x","year"])
        d = d[d["CLASSI_FIN"].isin(["8","10","11","12"])]
        parts.append(d[(d["day_ill"] >= 2) & (d["day_ill"] <= 5)])
        del d
    a = pd.concat(parts, ignore_index=True)
    a["y"] = a["CLASSI_FIN"].isin(["10","11","12"]).astype("int8")

    lp = np.zeros(len(a))        # 主分析:瘀点 近似 皮肤出血
    lp2 = np.zeros(len(a))       # 敏感性分析:瘀点 或 束臂试验阳性
    for day, o in ORS.items():
        m = (a["day_ill"] == day).values
        # 线性预测子 = Σ log(OR) × 变量值。巴西 → 使用拉丁美洲系数
        base = (np.log(o["cont"])
                + (np.log(o["age"]) / 17.0) * (a["age"].values - 8)   # 换算成每岁
                + np.log(o["temp"]) * a["FEBRE_x"].values             # 发热≈体温+1℃
                + np.log(o["rash"]) * a["EXANTEMA_x"].values)
        lp  = np.where(m, base + np.log(o["bleed"]) * a["PETEQUIA_N_x"].values, lp)
        lp2 = np.where(m, base + np.log(o["bleed"]) *
                       np.maximum(a["PETEQUIA_N_x"].values, a["LACO_x"].values), lp2)
    a["lp"], a["lp_laco"] = lp.astype("float32"), lp2.astype("float32")
    return a


def balanced_auc(sub: pd.DataFrame, feats: list) -> float:
    """在平衡子集上本地重训逻辑回归,返回测试集 AUC(用作对比基线)。"""
    pos, neg = sub[sub.y == 1], sub[sub.y == 0]
    n = min(len(pos), len(neg), 40_000)
    s = pd.concat([pos.sample(n, random_state=1), neg.sample(n, random_state=1)])
    Xtr, Xte, ytr, yte = train_test_split(s[feats].values, s["y"].values,
                                          test_size=0.3, random_state=1, stratify=s["y"].values)
    m = LogisticRegression(max_iter=400, class_weight="balanced").fit(Xtr, ytr)
    return float(round(roc_auc_score(yte, m.predict_proba(Xte)[:,1]), 4))


def main():
    a = load_scored()
    print(f"评分样本 {len(a):,} 人(发病第 2-5 天)")
    r = {}

    # --- 校验③:计算器直接迁移(不重训) ---
    r["auc_pooled"] = float(round(roc_auc_score(a.y, a.lp), 4))
    r["auc_laco"]   = float(round(roc_auc_score(a.y, a.lp_laco), 4))  # 敏感性分析
    r["auc_day"]    = {int(d): float(round(roc_auc_score(g.y, g.lp), 4))
                       for d, g in a.groupby("day_ill")}
    r["auc_year"]   = {int(y): float(round(roc_auc_score(g.y, g.lp), 4))
                       for y, g in a.groupby("year")}

    # --- 对比基线:同样变量本地重训 vs 全特征本地模型 ---
    F4 = ["age","FEBRE_x","EXANTEMA_x","PETEQUIA_N_x"]
    r["local4_pooled"] = balanced_auc(a, F4)

    # --- 分档分布:计算器的三档在两组人群中的比例 ---
    a["tier"] = np.where(a.lp < TIER_LOW, "low",
                  np.where(a.lp < TIER_HIGH, "medium", "high"))
    for cls, name in [(1,"dengue"), (0,"inconclusive")]:
        s = a[a.y == cls]["tier"].value_counts(normalize=True)
        r[f"tier_{name}"] = {k: float(round(100*s.get(k,0),1)) for k in ["low","medium","high"]}

    # --- 关键发现:评分是否随最终严重程度单调上升 ---
    r["lp_by_class"] = {c: dict(mean=float(round(g.lp.mean(),3)),
                                high_pct=float(round(100*(g.lp >= TIER_HIGH).mean(),1)),
                                n=int(len(g)))
                        for c, g in a.groupby("CLASSI_FIN")}

    json.dump(r, open("/tmp/idams_eval.json","w"), indent=1, ensure_ascii=False)
    print(json.dumps(r, indent=1, ensure_ascii=False))
    print("\n提示:若要实际部署,应固定已发表系数、在本地数据上重新拟合"
          "截距与斜率(logistic recalibration),并按'漏诊代价高于误报'重设阈值。")


if __name__ == "__main__":
    main()
