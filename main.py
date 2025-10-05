# mk_temperature_use_stationid_58563.py
# 依赖: pip install meteostat pandas numpy matplotlib pymannkendall
# 运行环境: Python 3.8+
from datetime import datetime
import os
import pandas as pd
import numpy as np
from meteostat import Point, Daily, Stations
import pymannkendall as mk
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ========== 参数（优先使用 station_id，如为 None 则用经纬度回退） ==========
cities = {
    # 已知 meteostat station id 示例（常用机场/气象站 id）
    'Beijing': {'station_id': '54511', 'lat':39.9042, 'lon':116.4074, 'alt':40},
    'Wuhan':   {'station_id': '57494', 'lat':30.5928, 'lon':114.3055, 'alt':23},
    # Ningbo: 已替换为你提供的 meteostat 站点 id = '58563'
    'Ningbo':  {'station_id': '58563', 'lat':29.8683, 'lon':121.5440, 'alt':10}
}

start = datetime(2015,1,1)
end   = datetime(2024,12,31)
START_YEAR = start.year
END_YEAR = end.year
YEARS = list(range(START_YEAR, END_YEAR+1))

outdir = 'mk_temperature_results'
os.makedirs(outdir, exist_ok=True)

# 用于收集所有城市的年度极值（便于绘箱线图）
annual_max_df = pd.DataFrame(index=YEARS)
annual_min_df = pd.DataFrame(index=YEARS)
mk_records = []

def safe_mk_original_test(arr):
    """兼容性封装：对传入 numpy array 做 pymannkendall.original_test，并以 dict 返回关键字段"""
    try:
        res = mk.original_test(arr, alpha=0.05)
    except Exception as e:
        print("pymannkendall original_test 运行异常：", e)
        return None
    out = {}
    for key in ('trend','p','z','tau','sen_slope','slope'):
        out[key] = getattr(res, key, None)
    return out

def fetch_daily_by_station(station_id):
    """优先用 station_id 抓取 Daily；返回 DataFrame 或 None"""
    try:
        data = Daily(station_id, start, end).fetch()
        if data is None or data.empty:
            return None
        # 标准化列名小写
        cols_low = {c.lower(): c for c in data.columns}
        data = data.rename(columns=cols_low)
        return data
    except Exception as e:
        print(f"通过 station_id={station_id} 抓取数据时发生异常：{e}")
        return None

def try_get_daily_by_point(lat, lon, alt=None):
    """通过 Point 获取 Daily，若无数据返回空 DataFrame"""
    try:
        p = Point(lat, lon, alt)
        data = Daily(p, start, end).fetch()
        return data
    except Exception as e:
        print("通过 Point 获取 Daily 数据时发生异常：", e)
        return pd.DataFrame()

def try_find_nearby_station_and_fetch(lat, lon, max_candidates=30):
    """
    当 station_id / Point 无数据时，查找附近 Meteostat 站点并尝试按站点抓取 Daily。
    返回 (dataframe, station_id_used, station_name) 或 (None, None, None)
    """
    try:
        stations = Stations()
        nearby = stations.nearby(lat, lon).fetch(max_candidates)
    except Exception as e:
        print("查找附近站点失败：", e)
        return None, None, None

    if nearby is None or nearby.empty:
        print("未找到附近站点。")
        return None, None, None

    print("附近候选站（前10）：")
    try:
        head = nearby.head(10)
        for idx, row in head.iterrows():
            name = row.get('name', '')
            dist_m = row.get('distance', None)
            dist_km = f"{(dist_m/1000):.1f} km" if dist_m is not None else ""
            inventory = row.get('inventory', None)
            print(f" - id={idx}, name={name}, dist={dist_km}, inventory={inventory}")
    except Exception:
        pass

    for station_id in nearby.index:
        try:
            st_data = Daily(station_id, start, end).fetch()
        except Exception as e:
            print(f"尝试抓取站点 {station_id} 时出现异常：{e}")
            continue
        if st_data is None or st_data.empty:
            continue
        # 标准化列名小写
        cols_low = {c.lower(): c for c in st_data.columns}
        st_data = st_data.rename(columns=cols_low)
        if ('tmax' in st_data.columns) or ('tmin' in st_data.columns):
            station_name = nearby.loc[station_id].get('name', '') if station_id in nearby.index else ''
            print(f"使用站点 {station_id} ({station_name}) 作为 {lat},{lon} 的数据来源。")
            return st_data, station_id, station_name
    print("所有候选站点均未返回含 tmax/tmin 的日数据。")
    return None, None, None

# ========== 主循环（优先使用 station_id）==========
for city, cinfo in cities.items():
    print(f"\nProcessing {city} ...")
    lat = cinfo.get('lat')
    lon = cinfo.get('lon')
    alt = cinfo.get('alt', None)
    station_id = cinfo.get('station_id', None)

    data = None
    used_station = None
    used_station_name = None

    # 若提供 station_id，先尝试用站点抓取
    if station_id:
        print(f"Trying station_id={station_id} for {city} ...")
        data = fetch_daily_by_station(station_id)
        if data is not None:
            used_station = station_id
            used_station_name = ''  # 若需要可查 Stations() 填充名称

    # 若 station_id 无效或未提供，回退到 Point 并尝试附近站点
    if data is None or data.empty:
        print(f"No usable data from station_id for {city}. Falling back to Point / nearby search.")
        # 先尝试 Point
        data = try_get_daily_by_point(lat, lon, alt)
        if data is None or data.empty:
            # Point 无数据 -> 查附近站点
            st_data, sid_used, sname = try_find_nearby_station_and_fetch(lat, lon, max_candidates=30)
            if st_data is not None:
                data = st_data
                used_station = sid_used
                used_station_name = sname
            else:
                print(f"最终未找到适用于 {city} 的日数据，跳过该城市。")
                continue
        else:
            used_station = 'Point'

    # 标准化列名为小写（以防万一）
    data_cols_map = {c: c.lower() for c in data.columns}
    data = data.rename(columns=data_cols_map)

    if ('tmax' not in data.columns) and ('tmin' not in data.columns):
        print(f"抓到的数据不包含 'tmax' 或 'tmin' 列（城市 {city}）。跳过。")
        continue

    # 只保留 tmax, tmin
    keep_cols = []
    if 'tmax' in data.columns:
        keep_cols.append('tmax')
    if 'tmin' in data.columns:
        keep_cols.append('tmin')
    data = data[keep_cols].astype(float, errors='ignore')

    # 确保时间索引
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
        else:
            print(f"无法识别时间索引（city={city}），跳过。")
            continue

    data = data.dropna(how='all')

    # 计算年度极端
    data['year'] = data.index.year
    ann_max = data.groupby('year')['tmax'].max().reindex(YEARS) if 'tmax' in data.columns else pd.Series(index=YEARS, dtype=float)
    ann_min = data.groupby('year')['tmin'].min().reindex(YEARS) if 'tmin' in data.columns else pd.Series(index=YEARS, dtype=float)

    # 保存每城市年度CSV
    df_ann = pd.concat([ann_max.rename('annual_max'), ann_min.rename('annual_min')], axis=1).reset_index().rename(columns={'index':'year'})
    df_ann.to_csv(f'{outdir}/annual_extremes_{city}.csv', index=False, float_format='%.2f')

    # 合并到总体表
    annual_max_df[city] = ann_max.values
    annual_min_df[city] = ann_min.values

    # MK 检验
    mk_max = safe_mk_original_test(ann_max.values) if ann_max.dropna().size >= 3 else None
    mk_min = safe_mk_original_test(ann_min.values) if ann_min.dropna().size >= 3 else None

    mk_records.append({
        'city': city,
        'n_years': int(np.sum(~np.isnan(ann_max.values))),
        'used_station': used_station if used_station else 'unknown',
        'used_station_name': used_station_name if used_station_name else '',
        'mk_max_result': mk_max,
        'mk_min_result': mk_min
    })

# 保存 MK 结果为 DataFrame（便于查看）
mk_out = []
for rec in mk_records:
    city = rec['city']
    for var in ['max','min']:
        res = rec[f'mk_{var}_result']
        if res is None:
            mk_out.append({
                'city':city,
                'variable':f'annual_{var}',
                'n':rec['n_years'],
                'used_station': rec.get('used_station'),
                'trend':None, 'p':None, 'z':None, 'tau':None, 'slope':None
            })
        else:
            mk_out.append({
                'city':city,
                'variable':f'annual_{var}',
                'n':rec['n_years'],
                'used_station': rec.get('used_station'),
                'trend': res.get('trend'),
                'p': res.get('p'),
                'z': res.get('z'),
                'tau': res.get('tau'),
                'slope': res.get('sen_slope') or res.get('slope') or None
            })
mk_df = pd.DataFrame(mk_out)
mk_df.to_csv(f'{outdir}/mk_results.csv', index=False, float_format='%.4f')

# ========== 绘图 ==========
plt.figure(figsize=(8,6))
data_to_plot = [annual_max_df[c].dropna().values for c in annual_max_df.columns]
plt.boxplot(data_to_plot, labels=annual_max_df.columns, showmeans=True)
plt.title(f'Annual max of daily Tmax ({START_YEAR}-{END_YEAR}) — Boxplot (cities)')
plt.ylabel('°C')
plt.xlabel('City')
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{outdir}/boxplot_annual_max.png', dpi=300)
plt.close()

plt.figure(figsize=(8,6))
data_to_plot = [annual_min_df[c].dropna().values for c in annual_min_df.columns]
plt.boxplot(data_to_plot, labels=annual_min_df.columns, showmeans=True)
plt.title(f'Annual min of daily Tmin ({START_YEAR}-{END_YEAR}) — Boxplot (cities)')
plt.ylabel('°C')
plt.xlabel('City')
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{outdir}/boxplot_annual_min.png', dpi=300)
plt.close()

print('\nDone. Outputs saved in folder:', outdir)
print('Files: annual_extremes_{city}.csv, mk_results.csv, boxplot_annual_max.png, boxplot_annual_min.png')
