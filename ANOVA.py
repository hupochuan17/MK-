# ===============================
# 单因素方差分析（ANOVA）
# ===============================
import pandas as pd
from scipy import stats

# 读取数据
file_path = r"F:\TEST\mk_temperature_results\annual_extremes_allcities.csv"
df = pd.read_csv(file_path)

# 分析年度最高温（annual_max）差异
groups_max = [df[df['city'] == c]['annual_max'].dropna() for c in df['city'].unique()]
anova_max = stats.f_oneway(*groups_max)

print("===== 年度最高温 ANOVA 结果 =====")
print(f"F值: {anova_max.statistic:.4f}, p值: {anova_max.pvalue:.4f}")

# 分析年度最低温（annual_min）差异
groups_min = [df[df['city'] == c]['annual_min'].dropna() for c in df['city'].unique()]
anova_min = stats.f_oneway(*groups_min)

print("\n===== 年度最低温 ANOVA 结果 =====")
print(f"F值: {anova_min.statistic:.4f}, p值: {anova_min.pvalue:.4f}")

# 判断显著性
if anova_max.pvalue < 0.05:
    print("\n→ 年度最高温在三地间存在显著差异。")
else:
    print("\n→ 年度最高温在三地间无显著差异。")

if anova_min.pvalue < 0.05:
    print("→ 年度最低温在三地间存在显著差异。")
else:
    print("→ 年度最低温在三地间无显著差异。")
