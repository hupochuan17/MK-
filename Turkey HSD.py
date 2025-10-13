# anova_pairwise_extremes.py
# 运行环境: Python 3.8+
# 依赖: pandas, numpy, scipy, statsmodels, scikit_posthocs, matplotlib, seaborn, pingouin (可选)
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

# 可选：用于效应量，若未安装，下方也提供手算函数
try:
    import pingouin as pg
    HAVE_PG = True
except Exception:
    HAVE_PG = False

sns.set(style="whitegrid", context="talk")

# ========== 配置 ==========
# 把这里的路径改为你保存 annual_extremes_*.csv 的文件夹
DATA_DIR = Path(r"F:\TEST\mk_temperature_results")   # <- 修改为你的文件夹
OUT_DIR = DATA_DIR / "anova_results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CITY_FILES = {
    "Beijing": DATA_DIR / "annual_extremes_Beijing.csv",
    "Wuhan" : DATA_DIR / "annual_extremes_Wuhan.csv",
    "Ningbo": DATA_DIR / "annual_extremes_Ningbo.csv"   # 你的本地文件
}

YEARS = list(range(2015, 2025))

# ========== 工具函数 ==========
def read_city_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    df = pd.read_csv(path)
    # 规范列名小写
    df.columns = [c.lower() for c in df.columns]
    # 期望有 year, annual_max, annual_min 或类似列
    # 自动识别极值列
    max_col = None
    min_col = None
    for c in df.columns:
        if "max" in c:
            max_col = c
        if "min" in c:
            min_col = c
    if max_col is None or min_col is None:
        raise ValueError(f"{path} 中没有包含 'max' 或 'min' 的列，请检查列名。当前列: {df.columns.tolist()}")
    # 保证 year 列存在
    if "year" not in df.columns:
        df = df.rename(columns={df.columns[0]: "year"})
    df = df.set_index("year")
    # 返回 Series indexed by year
    smax = df[max_col].reindex(YEARS)
    smin = df[min_col].reindex(YEARS)
    return smax.astype(float), smin.astype(float)

def eta_squared_anova(aov_table):
    # eta^2 = SS_between / SS_total
    # statsmodels ANOVA table usually has sum_sq column
    ss_between = aov_table.loc['C(group)', 'sum_sq'] if 'C(group)' in aov_table.index else aov_table.iloc[0]['sum_sq']
    ss_total = aov_table['sum_sq'].sum()
    return ss_between / ss_total

def cohens_d(x, y):
    # pooled sd
    nx = len(x)
    ny = len(y)
    sx = np.nanstd(x, ddof=1)
    sy = np.nanstd(y, ddof=1)
    pooled = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2))
    return (np.nanmean(x) - np.nanmean(y)) / pooled

# ========== 载入数据并整理成长表 ==========
records = []
for city, fpath in CITY_FILES.items():
    smax, smin = read_city_file(fpath)
    for yr in YEARS:
        records.append({"city": city, "year": yr, "tmax": smax.loc[yr] if yr in smax.index else np.nan, "tmin": smin.loc[yr] if yr in smin.index else np.nan})

df_long = pd.DataFrame(records)
# 保存合并表
df_long.to_csv(OUT_DIR / "combined_annual_extremes_long.csv", index=False)

# ========== 对 Tmax 和 Tmin 分别做分析 ==========
results_summary = []
for var in ["tmax", "tmin"]:
    print("\n" + "="*60)
    print(f"变量: {var}")
    # 准备数据（按组）
    df_drop = df_long[["city", "year", var]].dropna()
    groups = [group[var].values for name, group in df_drop.groupby("city")]

    # 1) 正态性检验（每组）
    shapiro_results = {}
    for name, group in df_drop.groupby("city"):
        if len(group[var].dropna()) >= 3:
            shapiro_results[name] = stats.shapiro(group[var].dropna())
        else:
            shapiro_results[name] = (np.nan, np.nan)
    print("Shapiro-Wilk per group:", shapiro_results)

    # 2) 方差齐性检验（Levene）
    try:
        levene_p = stats.levene(*groups, center='median').pvalue
    except Exception as e:
        levene_p = np.nan
    print("Levene p-value:", levene_p)

    # 3) 根据前提选择 ANOVA 或 Kruskal-Wallis
    normal_ok = all((not np.isnan(shapiro_results[c][1])) and (shapiro_results[c][1] > 0.05) for c in shapiro_results)
    homogeneity_ok = not np.isnan(levene_p) and (levene_p > 0.05)

    if normal_ok and homogeneity_ok:
        print("满足正态性与方差齐性 -> 进行单因素 ANOVA")
        # 使用 statsmodels 便于后续 Tukey
        model = ols(f"{var} ~ C(city)", data=df_drop).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        print(aov_table)
        # eta2
        eta2 = eta_squared_anova(aov_table)
        print("Eta-squared:", eta2)
        # Tukey HSD
        tukey = pairwise_tukeyhsd(df_drop[var], df_drop["city"], alpha=0.05)
        print(tukey.summary())
        # 保存结果
        aov_table.to_csv(OUT_DIR / f"aov_table_{var}.csv")
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukey_df.to_csv(OUT_DIR / f"tukey_{var}.csv", index=False)
        resdict = {"var": var, "method": "ANOVA", "aov": aov_table, "eta2": eta2, "tukey": tukey_df}
    else:
        print("不满足 ANOVA 前提 -> 使用 Kruskal-Wallis（非参数）")
        kw_stat, kw_p = stats.kruskal(*groups, nan_policy='omit')
        print("Kruskal-Wallis H:", kw_stat, "p:", kw_p)
        # 如果显著，做两两 Mann-Whitney (Bonferroni)
        pairwise = []
        cities = df_drop['city'].unique()
        for i in range(len(cities)):
            for j in range(i+1, len(cities)):
                a = df_drop[df_drop['city']==cities[i]][var].dropna()
                b = df_drop[df_drop['city']==cities[j]][var].dropna()
                if len(a) >= 3 and len(b) >= 3:
                    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                    p_adj = min(p * 3, 1.0)  # Bonferroni for 3 pairwise
                    d = cohens_d(a, b)
                    pairwise.append({"group1": cities[i], "group2": cities[j], "u": u, "p": p, "p_bonf": p_adj, "cohens_d": d})
        pairwise_df = pd.DataFrame(pairwise)
        if 'kw_stat' in locals():
            # save
            pd.DataFrame([{"kw_stat": kw_stat, "kw_p": kw_p}]).to_csv(OUT_DIR / f"kruskal_{var}.csv", index=False)
        pairwise_df.to_csv(OUT_DIR / f"pairwise_mannwhitney_{var}.csv", index=False)
        resdict = {"var": var, "method": "Kruskal", "kw_stat": kw_stat, "kw_p": kw_p, "pairwise": pairwise_df}

    results_summary.append(resdict)

# ========== 绘图：箱线图 + 标注显著性 ==========
# 简单箱线图（两个图：tmax / tmin）
for var in ["tmax", "tmin"]:
    plt.figure(figsize=(8,6))
    ax = sns.boxplot(x="city", y=var, data=df_long, showmeans=True, palette="Set2")
    sns.stripplot(x="city", y=var, data=df_long, color='k', size=6, jitter=True, ax=ax)
    plt.title(f"Annual {var.upper()} distribution (2015-2024)")
    plt.ylabel("Temperature (°C)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"boxplot_{var}.png", dpi=300)
    plt.close()

# ========== 保存汇总摘要 ==========
# 把 results_summary 中的主要数值写成 CSV（简单版）
summary_rows = []
for r in results_summary:
    if r['method'] == "ANOVA":
        aov = r['aov']
        summary_rows.append({
            "var": r['var'],
            "method": "ANOVA",
            "eta2": r['eta2'],
            "p_value": aov.loc['C(city)', 'PR(>F)'] if 'C(city)' in aov.index else aov.iloc[0]['PR(>F)']
        })
    else:
        summary_rows.append({
            "var": r['var'],
            "method": "Kruskal",
            "eta2": None,
            "p_value": r.get('kw_p', None)
        })
pd.DataFrame(summary_rows).to_csv(OUT_DIR / "anova_kruskal_summary.csv", index=False)

print("\n全部完成。结果文件夹：", OUT_DIR)
