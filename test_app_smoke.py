import unittest

import akshare as ak
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def fake_daily(symbol):
    dates = pd.bdate_range("2016-01-01", periods=2100)
    x = np.arange(len(dates), dtype=float)
    base = {
        "sh000300": 3500.0,
        "sh000905": 5200.0,
        "sh000852": 6100.0,
    }[symbol]
    slope = {"sh000300": 0.5, "sh000905": 0.65, "sh000852": 0.55}[symbol]
    close = base + slope * x + 80.0 * np.sin(x / 45.0)
    return pd.DataFrame({"date": dates, "close": close})


def fake_spot():
    return pd.DataFrame(
        {
            "代码": ["sh000300", "sh000905", "sh000852"],
            "最新价": [4550.0, 6500.0, 7200.0],
        }
    )


class DashboardSmokeTest(unittest.TestCase):
    def test_home_page_renders_with_synthetic_market_data(self):
        original_daily = ak.stock_zh_index_daily
        original_spot = ak.stock_zh_index_spot_sina
        ak.stock_zh_index_daily = fake_daily
        ak.stock_zh_index_spot_sina = fake_spot
        try:
            app = AppTest.from_file("streamlit_app.py")
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertIn("指数比值趋势偏离看板", app.markdown[1].value)
            self.assertEqual(len(app.button), 4)
            app.button[1].click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertIn("IC / IF 趋势偏离分析", app.markdown[1].value)
            self.assertEqual(app.radio[0].value, 5)
            self.assertTrue(any("完整比值从" in item.value for item in app.markdown))
            for years in (3, 7):
                app.radio[0].set_value(years).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.radio[0].value, years)
        finally:
            ak.stock_zh_index_daily = original_daily
            ak.stock_zh_index_spot_sina = original_spot


if __name__ == "__main__":
    unittest.main()
