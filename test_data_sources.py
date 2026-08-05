import unittest
from pathlib import Path

import pandas as pd

from data_sources import (
    exchange_ifind_access_token,
    fetch_ifind_history,
    load_embedded_history,
    merge_market_history,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payload)


class MarketDataTests(unittest.TestCase):
    def test_embedded_history_is_clean_and_complete(self):
        path = Path(__file__).parent / "data" / "index_history.csv"
        frame = load_embedded_history(path)

        self.assertEqual(frame.index.min(), pd.Timestamp("2002-01-04"))
        self.assertEqual(frame.index.max(), pd.Timestamp("2026-08-05"))
        self.assertEqual(frame["zz1000"].notna().sum(), 5244)
        self.assertEqual(frame["zz500"].notna().sum(), 5244)
        for invalid in ("2014-11-15", "2018-06-18", "2020-05-05", "2020-10-08"):
            self.assertNotIn(pd.Timestamp(invalid), frame.index)

    def test_refresh_exchange_sends_content_length_body(self):
        session = FakeSession(
            {
                "errorcode": 0,
                "data": {"access_token": "access", "expired_time": "2026-08-12 15:15:38"},
            }
        )
        token, expiry = exchange_ifind_access_token("refresh", session=session)

        self.assertEqual(token, "access")
        self.assertEqual(expiry, "2026-08-12 15:15:38")
        self.assertEqual(session.calls[0][1]["data"], b"")

    def test_ifind_history_parser_maps_all_indices(self):
        session = FakeSession(
            {
                "errorcode": 0,
                "tables": [
                    {"thscode": "000300.SH", "time": ["2026-08-05"], "table": {"close": [4658.1549]}},
                    {"thscode": "000905.SH", "time": ["2026-08-05"], "table": {"close": [7809.2065]}},
                    {"thscode": "000852.SH", "time": ["2026-08-05"], "table": {"close": [7493.0481]}},
                ],
            }
        )
        frame = fetch_ifind_history("access", "2026-08-05", "2026-08-05", session=session)

        self.assertAlmostEqual(frame.loc["2026-08-05", "hs300"], 4658.1549)
        self.assertAlmostEqual(frame.loc["2026-08-05", "zz500"], 7809.2065)
        self.assertAlmostEqual(frame.loc["2026-08-05", "zz1000"], 7493.0481)

    def test_merge_keeps_ifind_primary_and_uses_akshare_only_for_gaps(self):
        embedded = pd.DataFrame(
            {"hs300": [1.0], "zz500": [2.0], "zz1000": [3.0]},
            index=pd.to_datetime(["2026-08-03"]),
        )
        ifind = pd.DataFrame(
            {"hs300": [1.1, 1.2], "zz500": [2.1, 2.2], "zz1000": [3.1, 3.2]},
            index=pd.to_datetime(["2026-08-03", "2026-08-04"]),
        )
        fallback = pd.DataFrame(
            {"hs300": [9.0, 1.3], "zz500": [9.0, 2.3], "zz1000": [9.0, 3.3]},
            index=pd.to_datetime(["2026-08-03", "2026-08-05"]),
        )

        result, metadata = merge_market_history(embedded, ifind, fallback)

        self.assertAlmostEqual(result.loc["2026-08-03", "hs300"], 1.1)
        self.assertAlmostEqual(result.loc["2026-08-04", "zz500"], 2.2)
        self.assertAlmostEqual(result.loc["2026-08-05", "zz1000"], 3.3)
        self.assertEqual(metadata["fallback_dates"], 1)


if __name__ == "__main__":
    unittest.main()
