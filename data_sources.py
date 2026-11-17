"""Market-data loading with iFinD, CSI and dated Sina daily bars."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


INDEX_COLUMNS = ("hs300", "zz500", "zz1000")
IFIND_CODES = {
    "000300.SH": "hs300",
    "000905.SH": "zz500",
    "000852.SH": "zz1000",
}
CSI_HISTORY_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
SINA_DAILY_URL = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
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
    """Load the bundled history of verified index closes."""

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


def fetch_public_index(item, start, end, session=requests):
    code, column = item
    symbol = code.split(".")[0]
    closes = pd.Series(dtype=float, name=column, index=pd.DatetimeIndex([]))

    def dated_closes(rows, date_field):
        dates = pd.to_datetime(rows[date_field].astype(str), errors="coerce").dt.normalize()
        values = pd.to_numeric(rows["close"], errors="coerce")
        result = pd.Series(values.to_numpy(), index=dates, name=column).dropna()
        result = result.loc[(result.index >= start) & (result.index <= end)]
        result = result[(result > 0) & result.map(isfinite)]
        return result[~result.index.duplicated(keep="last")].sort_index()

    try:
        response = session.get(CSI_HISTORY_URL, params={
            "indexCode": symbol, "startDate": start.strftime("%Y%m%d"), "endDate": end.strftime("%Y%m%d")
        }, timeout=(18, 15))
        response.raise_for_status()
        rows = pd.DataFrame(response.json()["data"])
        closes = dated_closes(rows.loc[rows["indexCode"].eq(symbol)], "tradeDate")
    except (requests.RequestException, KeyError, TypeError, ValueError):
        pass

    # CSI publishes closing levels. Sina also serves dated bars during the session.
    if closes.empty or closes.index.max() < end:
        try:
            response = session.get(SINA_DAILY_URL, params={
                "symbol": "sh" + symbol, "scale": 240, "ma": "no",
                "datalen": (end - start).days + 1,
            }, timeout=(18, 15))
            response.raise_for_status()
            sina = dated_closes(pd.DataFrame(response.json()), "day")
            closes = closes.combine_first(sina)
        except (requests.RequestException, KeyError, TypeError, ValueError):
            pass
    return closes.rename(column)


def fetch_dated_quotes(today, session=requests):
    """Return the last trade time and current/previous index levels from Tencent."""
    if today.weekday() >= 5:
        return None
    symbols = {"sh" + code.split(".")[0]: column for code, column in IFIND_CODES.items()}
    try:
        response = session.get("https://qt.gtimg.cn/q=" + ",".join(symbols), timeout=(18, 10))
        response.raise_for_status()
        records = {}
        for line in response.content.decode("gbk").splitlines():
            name, _, payload = line.partition("=")
            symbol = name.removeprefix("v_")
            if symbol not in symbols:
                continue
            fields = payload.strip().strip('";').split("~")
            if fields[2] != symbol[2:]:
                continue
            timestamp = pd.to_datetime(fields[30], format="%Y%m%d%H%M%S")
            price, previous = float(fields[3]), float(fields[4])
            if timestamp.date() != today or timestamp.strftime("%H:%M") < "09:25":
                continue
            if all(isfinite(level) and level > 0 for level in (price, previous)):
                records[symbols[symbol]] = (price, previous, timestamp)
        if len(records) == len(symbols):
            return pd.DataFrame.from_dict(records, orient="index", columns=["price", "previous", "time"])
    except (requests.RequestException, UnicodeError, ValueError, IndexError):
        pass
    return None


def merge_market_history(
    embedded: pd.DataFrame,
    ifind: pd.DataFrame | None,
    public_history: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    """Merge sources without overwriting iFinD or bundled closes with public fallback values."""

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
    if public_history is not None and not public_history.empty:
        fallback = _normalise_frame(public_history)
        qa_start = combined.index.max() - pd.Timedelta(days=90)
        overlap = combined.loc[qa_start:].join(
            fallback.loc[qa_start:], how="inner", lsuffix="_official", rsuffix="_public"
        )
        differences = []
        for column in INDEX_COLUMNS:
            official_values = overlap[f"{column}_official"]
            fallback_values = overlap[f"{column}_public"]
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
    """Load audited history, then append iFinD closes and independently dated public quotes."""

    embedded = load_embedded_history(history_path)
    today = today or pd.Timestamp.now(tz=ZoneInfo("Asia/Shanghai")).date()
    start = max(embedded.index[-1] - pd.Timedelta(days=30), pd.Timestamp("2021-08-05"))
    end = pd.Timestamp(today)
    ifind = None
    token_expiry = None
    ifind_error = None

    try:
        token = access_token
        if refresh_token:
            token, token_expiry = exchange_ifind_access_token(refresh_token)
        if token:
            ifind = fetch_ifind_history(token, start, end)
    except Exception as exc:  # keep the dashboard available on upstream outages
        ifind_error = str(exc)

    with ThreadPoolExecutor(max_workers=len(IFIND_CODES)) as pool:
        requests_by_index = [
            pool.submit(fetch_public_index, item, start, end) for item in IFIND_CODES.items()
        ]
        public_history = pd.concat([job.result() for job in requests_by_index], axis=1)

    combined, metadata = merge_market_history(embedded.loc[:end], ifind, public_history)
    combined = combined.loc[:end]
    metadata["latest_date"] = combined.index.max()
    metadata.update(
        {
            "ifind_configured": bool(refresh_token or access_token),
            "ifind_error": ifind_error,
            "token_expiry": token_expiry,
            "embedded_latest_date": embedded.index[-1],
        }
    )
    return combined, metadata
