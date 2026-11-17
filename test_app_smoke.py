import unittest
from datetime import date

from unittest.mock import patch

from data_sources import load_embedded_history
from streamlit import cache_data
import pandas as pd
from streamlit.testing.v1 import AppTest


@patch("data_sources.fetch_dated_quotes", return_value=None)
class DashboardSmokeTest(unittest.TestCase):
    def setUp(self):
        cache_data.clear()

    def test_home_page_renders_with_bundled_market_data(self, _quotes):
        with patch("data_sources.load_market_history", return_value=(load_embedded_history(), {})):
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
            self.assertEqual(home_text.count('class="pair-head"'), 4)
            self.assertIn("BTC / 黄金", home_text)
            self.assertIn("19.8895", home_text)
            self.assertEqual(len(app.metric), 0)
            self.assertEqual(len(app.get("plotly_chart")), 4)
            self.assertTrue(any("BTC / 黄金 详情" in item.label for item in app.button))
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

    def test_im_ic_date_selection_and_refresh(self, _quotes):
        history = load_embedded_history()
        initial = history.loc[:"2026-08-05"]
        refreshed = history.loc[:"2026-11-16"].copy()
        refreshed.loc["2026-11-16", "zz1000"] = float("nan")
        today = pd.Timestamp.now(tz="Asia/Shanghai").date()
        with patch("data_sources.load_market_history", side_effect=[
            (initial, {}), (refreshed, {}), (initial, {}), (initial, {})
        ]) as fetch:
            app = AppTest.from_file("streamlit_app.py").run(timeout=30)
            app.button(key="detail_im_ic").click().run()
            self.assertEqual(app.date_input[0].value, date(2026, 8, 5))
            self.assertEqual(app.date_input[0].max, today)
            app.button(key="latest_im_ic").click().run()
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(app.date_input[0].value, date(2026, 11, 13))
            app.date_input[0].set_value(date(2026, 11, 16)).run()
            self.assertEqual(app.date_input[0].value, date(2026, 11, 16))
            self.assertIn("可用共同报价仅到 <strong>2026-11-13", " ".join(i.value for i in app.markdown))
            app.date_input[0].set_value(date(2026, 9, 2)).run()
            self.assertEqual(fetch.call_count, 3)
            self.assertIn("本页已切片到 2026-09-02", " ".join(i.value for i in app.markdown))
            app.date_input[0].set_value(app.date_input[0].min).run()
            self.assertTrue(any("不足 3 条" in item.value for item in app.info))
            app.button(key="latest_im_ic").click().run()
            self.assertEqual(app.date_input[0].value, date(2026, 11, 13))
            self.assertEqual(len(app.get("plotly_chart")), 8)
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
