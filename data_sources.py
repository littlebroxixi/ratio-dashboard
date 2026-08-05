"""Market-data loading with iFinD as primary and AKShare as fallback."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import requests


INDEX_COLUMNS = ("hs300", "zz500", "zz1000")
IFIND_CODES = {
    "000300.SH": "hs300",
    "000905.SH": "zz500",
    "000852.SH": "zz1000",
}
AKSHARE_SYMBOLS = {
    "sh000300": "hs300",
    "sh000905": "zz500",
    "sh000852": "zz1000",
}
IFIND_TOKEN_URL = "https://quantapi.51ifind.com/api/v1/get_access_token"
IFIND_HISTORY_URL = "https://quantapi.51ifind.com/api/v1/cmd_history_quotation"
DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent / "data" / "index_history.csv"


class MarketDataError(RuntimeError):
    """Raised when one upstream market-data request cannot be used."""


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.index).normalize()
    for column in INDEX_COLUMNS:
        if column not in result:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[:, list(INDEX_COLUMNS)]
    return result.loc[~result.index.duplicated(keep="last")].sort_index()


def load_embedded_history(path: str | Path = DEFAULT_HISTORY_PATH) -> pd.DataFrame:
    """Load the audited iFinD Excel history bundled with the dashboard."""

    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return _normalise_frame(frame)


def exchange_ifind_access_token(refresh_token: str, session=requests) -> tuple[str, str | None]:
    """Exchange a long-lived iFinD refresh token for an access token."""

    response = session.post(
        IFIND_TOKEN_URL,
        headers={"Content-Type": "application/json", "refresh_token": refresh_token},
        data=b"",
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errorcode", payload.get("errcode", -1)) != 0:
        raise MarketDataError(payload.get("errmsg") or "同花顺令牌续期失败")
    data = payload.get("data") or {}
    token = data.get("access_token")
    if not token:
        raise MarketDataError("同花顺令牌响应缺少 access_token")
    return token, data.get("expired_time")


def fetch_ifind_history(
    access_token: str,
    start_date,
    end_date,
    session=requests,
) -> pd.DataFrame:
    """Fetch official daily closes for the three monitored indices."""

    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    response = session.post(
        IFIND_HISTORY_URL,
        headers={"Content-Type": "application/json", "access_token": access_token},
        json={
            "codes": ",".join(IFIND_CODES),
            "indicators": "close",
            "startdate": start,
            "enddate": end,
            "functionpara": {"Fill": "Blank"},
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errorcode") != 0:
        raise MarketDataError(payload.get("errmsg") or "同花顺历史行情请求失败")

    series = []
    for table in payload.get("tables") or []:
        column = IFIND_CODES.get(table.get("thscode"))
        dates = table.get("time") or []
        closes = (table.get("table") or {}).get("close") or []
        if column and dates and len(dates) == len(closes):
            item = pd.DataFrame({column: closes}, index=pd.to_datetime(dates))
            series.append(item)
    if not series:
        return pd.DataFrame(columns=INDEX_COLUMNS, index=pd.DatetimeIndex([], name="date"))
    frame = pd.concat(series, axis=1)
    return _normalise_frame(frame)


def fetch_akshare_history() -> pd.DataFrame:
    """Fetch AKShare daily closes for validation and temporary gap filling."""

    series = []
    for symbol, column in AKSHARE_SYMBOLS.items():
        item = ak.stock_zh_index_daily(symbol=symbol)
        item["date"] = pd.to_datetime(item["date"])
        item = item.set_index("date")[["close"]].rename(columns={"close": column})
        series.append(item)
    return _normalise_frame(pd.concat(series, axis=1))


def merge_market_history(
    embedded: pd.DataFrame,
    ifind: pd.DataFrame | None,
    akshare: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    """Merge sources without allowing AKShare to overwrite official iFinD values."""

    embedded = _normalise_frame(embedded)
    combined = embedded.copy()
    ifind_rows = 0
    if ifind is not None and not ifind.empty:
        official = _normalise_frame(ifind)
        ifind_rows = int(official.dropna(how="all").shape[0])
        combined = official.combine_first(combined)

    fallback_cells = 0
    fallback_dates = 0
    max_difference_bp = None
    if akshare is not None and not akshare.empty:
        fallback = _normalise_frame(akshare)
        qa_start = combined.index.max() - pd.Timedelta(days=90)
        overlap = combined.loc[qa_start:].join(
            fallback.loc[qa_start:], how="inner", lsuffix="_official", rsuffix="_ak"
        )
        differences = []
        for column in INDEX_COLUMNS:
            official_values = overlap[f"{column}_official"]
            fallback_values = overlap[f"{column}_ak"]
            valid = official_values.notna() & fallback_values.notna() & fallback_values.ne(0)
            if valid.any():
                relative_bp = (
                    (official_values[valid] - fallback_values[valid]).abs()
                    / fallback_values[valid].abs()
                    * 10000
                )
                differences.append(relative_bp)
        if differences:
            max_difference_bp = float(pd.concat(differences).max())

        before = combined.copy()
        combined = combined.combine_first(fallback)
        before_aligned = before.reindex(index=combined.index, columns=combined.columns)
        filled = before_aligned.isna() & combined.notna()
        fallback_cells = int(filled.to_numpy().sum())
        fallback_dates = int(filled.any(axis=1).sum())

    combined = _normalise_frame(combined).dropna(how="all")
    metadata = {
        "ifind_rows": ifind_rows,
        "fallback_cells": fallback_cells,
        "fallback_dates": fallback_dates,
        "max_difference_bp": max_difference_bp,
        "latest_date": combined.index[-1] if len(combined) else None,
    }
    return combined, metadata


def load_market_history(
    refresh_token: str | None = None,
    access_token: str | None = None,
    today: date | None = None,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Load audited history, then append iFinD closes and AKShare fallbacks."""

    embedded = load_embedded_history(history_path)
    today = today or pd.Timestamp.now(tz=ZoneInfo("Asia/Shanghai")).date()
    start = max(embedded.index[-1] - pd.Timedelta(days=30), pd.Timestamp("2021-08-05"))
    end = pd.Timestamp(today)
    ifind = None
    akshare = None
    token_expiry = None
    ifind_error = None
    akshare_error = None

    try:
        token = access_token
        if refresh_token:
            token, token_expiry = exchange_ifind_access_token(refresh_token)
        if token:
            ifind = fetch_ifind_history(token, start, end)
    except Exception as exc:  # keep the dashboard available on upstream outages
        ifind_error = str(exc)

    try:
        akshare = fetch_akshare_history()
    except Exception as exc:  # the embedded official history still remains usable
        akshare_error = str(exc)

    combined, metadata = merge_market_history(embedded, ifind, akshare)
    metadata.update(
        {
            "ifind_configured": bool(refresh_token or access_token),
            "ifind_error": ifind_error,
            "akshare_error": akshare_error,
            "token_expiry": token_expiry,
            "embedded_latest_date": embedded.index[-1],
        }
    )
    return combined, metadata
