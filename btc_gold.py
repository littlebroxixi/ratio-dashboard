"""BTC/gold calendar-window models and repository-local research data.

Fit and snapshot match btc_gold_research.analysis. This module is self-contained
so Streamlit Cloud needs only this repository, not its parent workspace.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

YEARS = (1, 3, 5, 7)
METHODS = ("OLS", "均值标准差")
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_daily():
    data = pd.read_csv(DATA_DIR / "btc_gold_daily.csv", parse_dates=["date"]).set_index("date")
    if data.empty or data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("BTC / 黄金日线日期为空、重复或未排序")
    prices = data[["btc_open", "gold_open", "btc_close", "gold_close", "ratio"]]
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("BTC / 黄金日线价格无效")
    if not np.allclose(data.ratio, data.btc_close / data.gold_close, rtol=1e-10, atol=1e-10):
        raise ValueError("BTC / 黄金比值与美元双腿收盘价不符")
    return data


def load_research():
    results = json.loads((DATA_DIR / "btc_gold_research.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((DATA_DIR / "btc_gold_daily.csv").read_bytes()).hexdigest()
    if digest != results["metadata"]["daily_sha256"]:
        raise ValueError("研究统计与日线底表版本不一致，请一并更新")
    return results


def fit(values, method):
    y = np.asarray(values, float)
    n = len(y)
    if n < 3 or not np.isfinite(y).all():
        raise ValueError("At least 3 finite observations are required")
    if method == "OLS":
        x = np.arange(n, dtype=float)
        xc = x - x.mean()
        slope = float(xc @ (y - y.mean()) / (xc @ xc))
        center = y.mean() + slope * xc
        sigma = float(np.sqrt(np.sum((y - center) ** 2) / (n - 2)))
    elif method == "均值标准差":
        center = np.full(n, y.mean())
        sigma = float(np.std(y, ddof=1))
        slope = 0.0
    else:
        raise ValueError(method)
    z = np.full(n, np.nan) if sigma <= 1e-12 else (y - center) / sigma
    return center, sigma, z, slope


def snapshot(base, years=None, asof=None):
    cut = base.loc[:pd.Timestamp(asof)] if asof is not None else base
    if cut.empty:
        raise ValueError("No data at or before the observation date")
    end = cut.index[-1]
    if years is not None:
        start = end - pd.DateOffset(years=years)
        if cut.index[0] > start:
            raise ValueError(f"Insufficient history for {years} calendar years")
        cut = cut.loc[start:]
    rows, models = [], {}
    for method in METHODS:
        center, sigma, z, slope = fit(cut.ratio, method)
        m = cut.copy()
        m["center"], m["sigma"], m["z"] = center, sigma, z
        for level in (1, 2, 3):
            m[f"upper_{level}"] = center + level * sigma
            m[f"lower_{level}"] = center - level * sigma
        models[method] = m
        row = {"window": "全样本" if years is None else f"{years}年", "years": years,
               "method": method, "start": str(cut.index[0].date()),
               "end": str(end.date()), "n": len(cut), "ratio": float(cut.ratio.iloc[-1]),
               "center": float(center[-1]), "sigma": sigma, "z": float(z[-1]),
               "slope_per_observation": slope}
        for level in (1, 2, 3):
            row[f"lower_{level}"] = float(center[-1] - level * sigma)
            row[f"upper_{level}"] = float(center[-1] + level * sigma)
        rows.append(row)
    return rows, models


