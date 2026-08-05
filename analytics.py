"""Fixed-window analytics for index-ratio monitoring.

Each 3/5/7-year view fits one trend to the latest fixed sample.  The regression
residual scale is the standard OLS residual error:

    sigma = sqrt(SSE / (n - 2))

where the two fitted parameters are the intercept and slope.

For comparison, the same sample is also evaluated with a constant arithmetic
mean and the ordinary sample standard deviation (ddof=1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252
WINDOW_YEARS = (3, 5, 7)


@dataclass(frozen=True)
class Consensus:
    direction: int
    confirmation_count: int
    strong_count: int
    label: str


def slice_asof(frame: pd.DataFrame, requested_date) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Cut a time series at the latest available date on or before a request."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame index must be a DatetimeIndex")
    ordered = frame.sort_index()
    requested = pd.Timestamp(requested_date).normalize()
    eligible = ordered.index[ordered.index.normalize() <= requested]
    if len(eligible) == 0:
        raise ValueError("requested date is before the first observation")
    actual_date = eligible[-1]
    return ordered.loc[:actual_date].copy(), actual_date


def fixed_ols_signal(base: pd.DataFrame, years: int) -> pd.DataFrame:
    """Fit one OLS trend to the latest fixed N-year sample.

    The returned trend, residual scale and Z-scores all use the same final
    ``years * 252`` observations.  This is a current cross-window snapshot,
    not a point-in-time backtest signal for earlier dates in the sample.
    """

    if years <= 0:
        raise ValueError("years must be positive")
    required = {"ratio", "price_a", "price_b"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    window = years * TRADING_DAYS
    frame = base.sort_index().tail(window).copy()
    if len(frame) < window:
        raise ValueError(f"{years}y model needs at least {window} observations")

    y = frame["ratio"].to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    center = intercept + slope * x
    residual = y - center
    residual_sigma = np.sqrt(np.sum(residual**2) / (window - 2))

    frame["center"] = center
    frame["residual_sigma"] = residual_sigma
    frame["slope_year"] = slope * TRADING_DAYS
    frame["residual"] = residual
    frame["zscore"] = residual / residual_sigma
    for level in (1, 2, 3):
        frame[f"upper_{level}"] = center + level * residual_sigma
        frame[f"lower_{level}"] = center - level * residual_sigma
    return frame


def fixed_mean_signal(base: pd.DataFrame, years: int) -> pd.DataFrame:
    """Evaluate the latest fixed N-year sample around its arithmetic mean.

    The center and standard deviation are fixed across the displayed sample.
    The ordinary sample standard deviation uses ``N - 1`` in the denominator.
    """

    if years <= 0:
        raise ValueError("years must be positive")
    required = {"ratio", "price_a", "price_b"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    window = years * TRADING_DAYS
    frame = base.sort_index().tail(window).copy()
    if len(frame) < window:
        raise ValueError(f"{years}y model needs at least {window} observations")

    y = frame["ratio"].to_numpy(dtype=float)
    center = float(np.mean(y))
    sigma = float(np.std(y, ddof=1))
    frame["center"] = center
    frame["residual_sigma"] = sigma
    frame["slope_year"] = 0.0
    frame["residual"] = y - center
    frame["zscore"] = frame["residual"] / sigma
    for level in (1, 2, 3):
        frame[f"upper_{level}"] = center + level * sigma
        frame[f"lower_{level}"] = center - level * sigma
    return frame


def build_pair_models(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    windows: tuple[int, ...] = WINDOW_YEARS,
) -> dict:
    """Build raw ratio plus OLS and mean/std fixed-window models for a pair."""

    base = pd.DataFrame(index=df.index)
    base["ratio"] = df[numerator] / df[denominator]
    base["price_a"] = df[numerator]
    base["price_b"] = df[denominator]
    base = base.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    models = {years: fixed_ols_signal(base, years) for years in windows}
    mean_models = {years: fixed_mean_signal(base, years) for years in windows}
    return {"base": base, "models": models, "mean_models": mean_models}


def consensus_snapshot(models: dict[int, pd.DataFrame]) -> Consensus:
    """Summarize how many windows confirm the same ±2σ direction."""

    latest_z = np.array([models[years].iloc[-1]["zscore"] for years in sorted(models)])
    non_zero = latest_z[np.abs(latest_z) >= 0.25]
    if len(non_zero) == 0:
        return Consensus(0, 0, 0, "正常波动")

    direction = int(np.sign(np.median(non_zero)))
    same_direction = np.sign(latest_z) == direction
    confirmation_count = int(np.sum(same_direction & (np.abs(latest_z) >= 2.0)))
    strong_count = int(np.sum(same_direction & (np.abs(latest_z) >= 2.5)))

    if np.any(np.sign(latest_z) != direction) and np.max(np.abs(latest_z)) >= 1.0:
        label = "周期分歧"
    elif confirmation_count == 3:
        label = "三周期共振"
    elif confirmation_count == 2:
        label = "双周期确认"
    elif confirmation_count == 1:
        label = "单周期预警"
    elif np.max(np.abs(latest_z)) >= 1.0:
        label = "进入观察"
    else:
        label = "正常波动"
    return Consensus(direction, confirmation_count, strong_count, label)
