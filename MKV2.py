# mk_temperature_combine_localNingbo.py
# 依赖: pip install meteostat pandas numpy matplotlib pymannkendall
# 运行环境: Python 3.8+
from datetime import datetime
import os
import pandas as pd
import numpy as np
from meteostat import Point, Daily
import pymannkendall as mk
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ========== 参数设置 ==========
cities = {
    'Beijing': {'station_id': '54511', 'lat': 39.9042, 'lon': 116.4074, 'alt': 40},
    'Wuhan':   {'station_id': '57494', 'lat': 30.5928, 'lon': 114.3055, 'alt': 23},
    'Ningbo':  {'station_id': '58563', 'lat': 29.8683, 'lon': 121.5440, 'alt': 10}
}

start = datetime(2015, 1, 1)
end   = datetime(2024, 12, 31)
YEARS = list(range(start.year, end.year + 1))

outdir = r"F:\TEST\mk_temperature_results"
os.makedirs(outdir, exist_ok=True)

# ✅ 宁波本地文件路径
local_ningbo_path = os.path.join(outdir, "annual_extremes_Ningbo.csv")

annual_max_df = pd.DataFrame(index=YEARS)
annual_min_df = pd.DataFrame(index=YEARS)
mk_records = []

# ========== MK检验函数 ==========
def safe_mk_test(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return {"trend": None, "p": None, "z": None, "tau": None, "slope": None}
    res = mk.original_test(arr, alpha=0.05)
    return {
        "trend": res.trend,
        "p": res.p,
        "z": res.z,
        "tau": res.Tau,
        "slope": getattr(res, 'slope', getattr(res, 'sen_slope', None))
    }

# ========== 主循环 ==========
for city, info in cities.items():
    print(f"\n=== Processing {city} ===")

    # —— 宁波使用本地极值文件 —— #
    if city == "Ningbo" and os.path.exists(local_ningbo_path):
        df_ann = pd.read_csv(local_ningbo_path)
        # 自动识别列名
        if "year" not in df_ann.columns:
            df_ann.rename(columns={df_ann.columns[0]: "year"}, inplace=True)
        df_ann.set_index("year", inplace=True)

        ann_max = df_ann["annual_max"].reindex(YEARS)
        ann_min = df_ann["annual_min"].reindex(YEARS)

        annual_max_df[city] = ann_max
        annual_min_df[city] = ann_min

        mk_max = safe_mk_test(ann_max.values)
        mk_min = safe_mk_test(ann_min.values)

        mk_records.append({
            "city": city,
            "n_years": len(ann_max.dropna()),
            "used_station": "local_file",
            "used_station_name": "宁波市气候公报 / 栎社观测",
            "mk_max_result": mk_max,
            "mk_min_result": mk_min
        })
        print(f"✅ 宁波：本地极值数据已导入，共 {len(ann_max.dropna())} 年。")
        continue

    # —— 其他城市使用 Meteostat —— #
    station_id = info["station_id"]
    try:
        data = Daily(station_id, start, end).fetch()
    except Exception as e:
        print(f"⚠️ {city} 数据抓取失败: {e}")
        continue

    if data.empty:
        print(f"⚠️ {city}：无有效数据，跳过。")
        continue

    data["year"] = data.index.year
    ann_max = data.groupby("year")["tmax"].max().reindex(YEARS)
    ann_min = data.groupby("year")["tmin"].min().reindex(YEARS)

    annual_max_df[city] = ann_max
    annual_min_df[city] = ann_min

    mk_max = safe_mk_test(ann_max.values)
    mk_min = safe_mk_test(ann_min.values)

    mk_records.append({
        "city": city,
        "n_years": len(ann_max.dropna()),
        "used_station": station_id,
        "used_station_name": "",
        "mk_max_result": mk_max,
        "mk_min_result": mk_min
    })
    print(f"✅ {city} 年度极值提取完成，共 {len(ann_max.dropna())} 年。")

# ========== 输出MK结果 ==========
mk_out = []
for rec in mk_records:
    for var in ['max', 'min']:
        res = rec[f"mk_{var}_result"]
        mk_out.append({
            "city": rec["city"],
            "variable": f"annual_{var}",
            "n": rec["n_years"],
            "used_station": rec["used_station"],
            "trend": res.get("trend"),
            "p": res.get("p"),
            "z": res.get("z"),
            "tau": res.get("tau"),
            "slope": res.get("slope")
        })
mk_df = pd.DataFrame(mk_out)
mk_df.to_csv(os.path.join(outdir, "mk_results.csv"), index=False, float_format="%.4f")

# ========== 绘制箱线图 ==========
plt.figure(figsize=(8,6))
plt.boxplot([annual_max_df[c].dropna() for c in annual_max_df.columns],
            labels=annual_max_df.columns, showmeans=True)
plt.title("Annual Max Temperature (2015–2024)")
plt.ylabel("Temperature (°C)")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "boxplot_annual_max.png"), dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.boxplot([annual_min_df[c].dropna() for c in annual_min_df.columns],
            labels=annual_min_df.columns, showmeans=True)
plt.title("Annual Min Temperature (2015–2024)")
plt.ylabel("Temperature (°C)")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "boxplot_annual_min.png"), dpi=300)
plt.close()

print("\n✅ 全部完成，结果保存在：", outdir)
print("输出文件：")
print(" - mk_results.csv")
print(" - boxplot_annual_max.png")
print(" - boxplot_annual_min.png")
