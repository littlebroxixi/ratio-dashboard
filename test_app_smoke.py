import unittest
from datetime import date

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
            home_text = " ".join(item.value for item in app.markdown)
            self.assertNotIn("等待至少两个窗口超过", home_text)
            self.assertNotIn("正式历史与每日收盘", home_text)
            self.assertNotIn("数据截至", home_text)
            self.assertNotIn("查看全部固定窗口", home_text)
            self.assertEqual(len(app.button), 5)
            self.assertIn("BTC / 黄金", [item.value for item in app.subheader])
            self.assertIn("19.88946", [item.value for item in app.metric])
            self.assertEqual(len(app.get("plotly_chart")), 4)
            self.assertTrue(any("BTC / 黄金完整分析" in item.label for item in app.button))
            next(button for button in app.button if button.label == "打开 IC / IF 详情").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertIn("IC / IF 固定窗口分析", app.markdown[1].value)
            self.assertEqual(len(app.radio), 0)
            detail_text = " ".join(item.value for item in app.markdown)
            self.assertIn('class="metric-item metric-pair"', detail_text)
            for label in ("1年固定窗口", "3年固定窗口", "5年固定窗口", "7年固定窗口"):
                self.assertIn(label, detail_text)
            self.assertIn("OLS趋势＋残差标准差", detail_text)
            self.assertIn("平均值＋普通标准差", detail_text)
            self.assertEqual(len(app.get("plotly_chart")), 8)
            self.assertEqual(len(app.date_input), 1)
            button_labels = {button.label for button in app.button}
            for label in ("回到最新", "1年", "3年", "5年", "7年", "全局"):
                self.assertIn(label, button_labels)
            for label in ("10年", "15年", "20年"):
                self.assertIn(label, button_labels)

            app.date_input[0].set_value(date(2023, 6, 10)).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            sliced_text = " ".join(item.value for item in app.markdown)
            self.assertIn("2023-06-09", sliced_text)
            self.assertIn("之后的数据完全不参与计算", sliced_text)

            next(button for button in app.button if button.label == "1年").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("plotly_chart")), 6)
            hidden_text = " ".join(item.value for item in app.markdown)
            self.assertNotIn("1年固定窗口", hidden_text)

            next(button for button in app.button if button.label == "全局").click().run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("plotly_chart")), 8)
            global_text = " ".join(item.value for item in app.markdown)
            self.assertIn("全局窗口", global_text)
        finally:
            ak.stock_zh_index_daily = original_daily
            ak.stock_zh_index_spot_sina = original_spot


if __name__ == "__main__":
    unittest.main()
