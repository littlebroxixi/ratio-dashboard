"""Rolling OLS analytics for index-ratio monitoring.

Each signal at date t is estimated only from observations before t.  The
regression residual scale is the standard OLS residual error:

    sigma = sqrt(SSE / (n - 2))

where the two fitted parameters are the intercept and slope.
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


def rolling_ols_signal(base: pd.DataFrame, years: int) -> pd.DataFrame:
    """Return one-step-ahead rolling OLS signals for a ratio DataFrame.

    ``base`` must contain ``ratio``, ``price_a`` and ``price_b``.  At each date
    the preceding ``years * 252`` observations are fitted, excluding the
    evaluation date itself.
    """

    if years <= 0:
        raise ValueError("years must be positive")
    required = {"ratio", "price_a", "price_b"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    frame = base.sort_index().copy()
    y = frame["ratio"].to_numpy(dtype=float)
    length = len(y)
    window = years * TRADING_DAYS
    if length <= window:
        raise ValueError(f"{years}y model needs more than {window} observations")

    global_x = np.arange(length, dtype=float)
    prefix_y = np.concatenate(([0.0], np.cumsum(y)))
    prefix_yy = np.concatenate(([0.0], np.cumsum(y * y)))
    prefix_iy = np.concatenate(([0.0], np.cumsum(global_x * y)))

    center = np.full(length, np.nan)
    sigma = np.full(length, np.nan)
    slope_year = np.full(length, np.nan)

    n = float(window)
    sum_x = n * (n - 1.0) / 2.0
    sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0
    x_mean = (n - 1.0) / 2.0
    sxx = n * (n * n - 1.0) / 12.0

    for end in range(window, length):
        start = end - window
        sum_y = prefix_y[end] - prefix_y[start]
        sum_yy = prefix_yy[end] - prefix_yy[start]
        sum_global_iy = prefix_iy[end] - prefix_iy[start]
        sum_xy = sum_global_iy - start * sum_y

        mean_y = sum_y / n
        slope = (sum_xy - x_mean * sum_y) / sxx
        intercept = mean_y - slope * x_mean
        predicted = intercept + slope * n

        sse = (
            sum_yy
            - 2.0 * intercept * sum_y
            - 2.0 * slope * sum_xy
            + n * intercept * intercept
            + 2.0 * intercept * slope * sum_x
            + slope * slope * sum_x2
        )
        residual_sigma = np.sqrt(max(0.0, sse) / (window - 2))

        center[end] = predicted
        sigma[end] = residual_sigma
        slope_year[end] = slope * TRADING_DAYS

    frame["center"] = center
    frame["residual_sigma"] = sigma
    frame["slope_year"] = slope_year
    frame["residual"] = frame["ratio"] - frame["center"]
    frame["zscore"] = frame["residual"] / frame["residual_sigma"]
    for level in (1, 2, 3):
        frame[f"upper_{level}"] = frame["center"] + level * frame["residual_sigma"]
        frame[f"lower_{level}"] = frame["center"] - level * frame["residual_sigma"]

    return frame.dropna(subset=["center", "residual_sigma", "zscore"])


def build_pair_models(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    windows: tuple[int, ...] = WINDOW_YEARS,
) -> dict:
    """Build the raw ratio and rolling OLS models for an index pair."""

    base = pd.DataFrame(index=df.index)
    base["ratio"] = df[numerator] / df[denominator]
    base["price_a"] = df[numerator]
    base["price_b"] = df[denominator]
    base = base.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    models = {years: rolling_ols_signal(base, years) for years in windows}
    return {"base": base, "models": models}


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
