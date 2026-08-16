#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_prepare_data.py — 特征工程:把 SINAN 原始通报表转成建模用的数值表
================================================================
输入:巴西 SINAN 年度登革热通报 CSV(DENGBR23/24/25.csv,121 列)
输出:/tmp/eng_{year}.parquet(26 个特征 + 结局 + 年份)

用法:
    python3 01_prepare_data.py 2023 /path/to/DENGBR23.csv
    # 大文件(如 2024 年 656 万行)内存不足时用分片版:
    python3 01_prepare_data.py 2024 /path/to/DENGBR24.csv --chunked

数据说明:
  SINAN = 巴西国家法定传染病通报系统(Sistema de Informação de Agravos de Notificação)
  下载:https://datasus.saude.gov.br/  (DENGBR = Dengue Brasil)
"""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 变量定义
# 14 个症状(SINAN 编码:1=有, 2=无, 9=未知)
SYMPT = ["FEBRE",      # 发热
         "MIALGIA",    # 肌痛
         "CEFALEIA",   # 头痛
         "EXANTEMA",   # 皮疹
         "VOMITO",     # 呕吐
         "NAUSEA",     # 恶心
         "DOR_COSTAS", # 背痛
         "CONJUNTVIT", # 结膜炎
         "ARTRITE",    # 关节炎
         "ARTRALGIA",  # 关节痛
         "PETEQUIA_N", # 瘀点(皮肤出血)
         "LEUCOPENIA", # 白细胞减少  ← 重症最强预测因子
         "LACO",       # 束臂试验阳性
         "DOR_RETRO"]  # 眼后痛
# 7 个合并症
COMORB = ["DIABETES",   # 糖尿病
          "HEMATOLOG",  # 血液病
          "HEPATOPAT",  # 肝病
          "RENAL",      # 肾病
          "HIPERTENSA", # 高血压
          "ACIDO_PEPT", # 消化性溃疡
          "AUTO_IMUNE"] # 自身免疫病
# 其他需要的原始列
OTHER = ["NU_IDADE_N",  # 年龄(编码值,非岁数)
         "CS_SEXO",     # 性别
         "DT_NOTIFIC",  # 通报日期
         "DT_SIN_PRI",  # 症状开始日期
         "SEM_PRI",     # 流行病学周
         "CLASSI_FIN"]  # 最终分类(结局变量)
USECOLS = SYMPT + COMORB + OTHER


def engineer(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """把原始编码字段转成模型可用的数值特征。"""
    out = pd.DataFrame()

    # (1) 症状/合并症 → 二值。SINAN 用 1=有、2=无、9=未知,只有 "1" 记为 1
    for c in SYMPT + COMORB:
        out[c + "_x"] = (df[c] == "1").astype("int8")

    # (2) 年龄解码。NU_IDADE_N 是编码而非岁数:
    #     4xxx = 岁(4031 → 31 岁);< 4000 = 小时/天/月(未满 1 岁)→ 记 0 岁
    age = pd.to_numeric(df["NU_IDADE_N"], errors="coerce")
    age = np.where((age >= 4000) & (age < 5000), age - 4000,
                   np.where(age < 4000, 0, np.nan))
    out["age"] = age.astype("float32")

    # (3) 性别:女=1,男=0,其余缺失
    out["sex_f"] = np.where(df["CS_SEXO"] == "F", 1.0,
                     np.where(df["CS_SEXO"] == "M", 0.0, np.nan)).astype("float32")

    # (4) 病程天数 = 通报日期 − 症状开始日期
    #     ⚠ 关键陷阱:SINAN 日期是 ISO "%Y-%m-%d",不是 "%d/%m/%Y"。
    #        用错格式会导致全部解析失败 → 样本清零(本项目实际踩过)
    dtn = pd.to_datetime(df["DT_NOTIFIC"], format="%Y-%m-%d", errors="coerce")
    dts = pd.to_datetime(df["DT_SIN_PRI"], format="%Y-%m-%d", errors="coerce")
    out["day_ill"] = ((dtn - dts).dt.days).astype("float32")

    # (5) 季节性:流行病学周(1-52)做正弦/余弦周期编码,
    #     使第 52 周与第 1 周在特征空间中相邻(而非数值相差 51)
    wk = pd.to_numeric(df["SEM_PRI"].str[-2:], errors="coerce").fillna(26)
    out["wk_sin"] = np.sin(2 * np.pi * wk / 52).astype("float32")
    out["wk_cos"] = np.cos(2 * np.pi * wk / 52).astype("float32")

    # (6) 结局变量:8=不确定/排除, 10=普通登革热, 11=伴警示症状, 12=重症
    out["CLASSI_FIN"] = df["CLASSI_FIN"].str.strip().str.replace(r"\.0$", "", regex=True)
    out["year"] = year

    # (7) 质量过滤
    out = out[(out["age"] >= 0) & (out["age"] <= 110)]      # 合理年龄
    out = out[(out["day_ill"] >= 0) & (out["day_ill"] <= 14)]  # 合理病程(负值/超长=录入错误)
    out = out.dropna(subset=["sex_f", "age"])
    return out


def main():
    year, path = int(sys.argv[1]), sys.argv[2]
    chunked = "--chunked" in sys.argv

    if chunked:
        # 大文件分片处理,避免内存耗尽
        parts, total = [], 0
        for chunk in pd.read_csv(path, dtype=str, usecols=USECOLS, chunksize=400_000):
            total += len(chunk)
            parts.append(engineer(chunk, year))
            print(f"  处理中… 累计原始行 {total:,}", flush=True)
        res = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(path, dtype=str, usecols=USECOLS)
        print(f"原始行数 {len(df):,}", flush=True)
        res = engineer(df, year)

    res.to_parquet(f"/tmp/eng_{year}.parquet")
    print(f"{year} 清洗后 {len(res):,} 条 | 结局分布 "
          f"{res['CLASSI_FIN'].value_counts().head().to_dict()}")


if __name__ == "__main__":
    main()
