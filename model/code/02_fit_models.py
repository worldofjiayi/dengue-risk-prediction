#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_fit_models.py — 拟合三个逻辑回归模型
=======================================
输入:/tmp/eng_2023.parquet, eng_2024.parquet, eng_2025.parquet(由 01 生成)
输出:/tmp/full_results.json(指标+系数)、ROC 与系数图的中间 pickle

三个互补的预测问题:
  模型 A  —— 是不是登革热?  确诊(10/11/12) vs 不确定(8)
              ⚠ SINAN 无真正的"非登革热"对照,缺真阴性
  模型 B  —— 会不会加重?    警示+重症(11/12) vs 普通(10)
  模型 B2 —— 会不会到最重症?  重症(12) vs 其余(10/11)

用法:
    python3 02_fit_models.py A     # 或 B / B2
"""
import sys, os, json, pickle, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score

WD = "/tmp"
SYMPT = ["FEBRE","MIALGIA","CEFALEIA","EXANTEMA","VOMITO","NAUSEA","DOR_COSTAS",
         "CONJUNTVIT","ARTRITE","ARTRALGIA","PETEQUIA_N","LEUCOPENIA","LACO","DOR_RETRO"]
COMORB = ["DIABETES","HEMATOLOG","HEPATOPAT","RENAL","HIPERTENSA","ACIDO_PEPT","AUTO_IMUNE"]
FEATS = [c+"_x" for c in SYMPT] + [c+"_x" for c in COMORB] + \
        ["age","sex_f","day_ill","wk_sin","wk_cos"]          # 共 26 个特征

NAMES = {"A":  "模型A:登革热 vs 不确定(无真阴性)",
         "B":  "模型B:警示+重症(11+12) vs 普通(10)",
         "B2": "模型B2:重症(12) vs 其他登革热(10+11)"}


def load_all() -> pd.DataFrame:
    """合并三年数据,只保留有明确结局的记录。"""
    parts = []
    for f in ["/tmp/eng_2023.parquet", "/tmp/eng_2024.parquet", "/tmp/eng_2025.parquet"]:
        d = pd.read_parquet(f)
        parts.append(d[d["CLASSI_FIN"].isin(["8","10","11","12"])])
        del d
    return pd.concat(parts, ignore_index=True)


def make_subset(a: pd.DataFrame, which: str) -> pd.DataFrame:
    """定义正负类,并对多数类做下采样(undersampling)。

    为什么要下采样:重症(12)仅占确诊病例约 0.15%。若直接训练,
    模型"永远预测非重症"就能拿到 99.8% 准确率,却一个重症都抓不到。
    做法:少数类全部保留,多数类随机抽样缩减 —— 既让模型认真对待
    少数类,也大幅加快训练。
    """
    if which == "A":
        a["y"] = a["CLASSI_FIN"].isin(["10","11","12"]).astype(int)
        pos, neg = a[a.y == 1], a[a.y == 0]
        return pd.concat([pos.sample(min(len(pos), 150_000), random_state=1),
                          neg.sample(min(len(neg), 150_000), random_state=1)])
    a = a[a["CLASSI_FIN"].isin(["10","11","12"])].copy()      # 仅确诊病例
    if which == "B":
        a["y"] = a["CLASSI_FIN"].isin(["11","12"]).astype(int)
    else:                                                     # B2
        a["y"] = (a["CLASSI_FIN"] == "12").astype(int)
    return pd.concat([a[a.y == 1],
                      a[a.y == 0].sample(min(200_000, int((a.y == 0).sum())), random_state=1)])


def main():
    which = sys.argv[1]
    s = make_subset(load_all(), which)
    y, X = s["y"].values, s[FEATS].values

    # 70/30 分层拆分:保证训练集与测试集的正负比例一致
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # class_weight="balanced":在下采样之外再给少数类更高权重,
    # 公式为 每类权重 = 总样本数 / (类别数 × 该类样本数)
    m = LogisticRegression(max_iter=600, class_weight="balanced").fit(Xtr, ytr)

    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    auc  = roc_auc_score(yte, p)
    sens = recall_score(yte, pred)                  # 敏感度:少漏诊
    spec = recall_score(yte, pred, pos_label=0)     # 特异度:少误报
    coef = pd.Series(m.coef_[0], index=FEATS).sort_values(ascending=False)

    print(f"===== {NAMES[which]} =====")
    print(f"n={len(y):,}  正例={int(y.sum()):,} ({100*y.mean():.2f}%)")
    print(f"AUC={auc:.3f}  敏感度={sens:.3f}  特异度={spec:.3f}")
    print("最强正向:", {k: round(v,2) for k,v in coef.head(6).items()})
    print("最强负向:", {k: round(v,2) for k,v in coef.tail(4).items()})

    r = dict(name=NAMES[which], n=int(len(y)), pos=int(y.sum()),
             auc=round(float(auc),4), sens=round(float(sens),4), spec=round(float(spec),4),
             coef={k: round(float(v),3) for k,v in coef.items()},
             cm=confusion_matrix(yte, pred).tolist())
    rp = f"{WD}/full_results.json"
    res = json.load(open(rp)) if os.path.exists(rp) else {}
    res[which] = r
    json.dump(res, open(rp, "w"), ensure_ascii=False, indent=2)

    if which in ("A","B"):
        pickle.dump({"p": p, "y": yte}, open(f"{WD}/_rocfull_{which}.pkl","wb"))
    if which == "B":
        coef.to_pickle(f"{WD}/_coefBfull.pkl")
    print("已保存", which)


if __name__ == "__main__":
    main()
