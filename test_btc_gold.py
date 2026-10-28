"""Deployment regression tests: only repository-local assets are required."""
from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest
from streamlit.runtime.pages_manager import PagesManager

import btc_gold
from test_app_smoke import fake_daily, fake_spot


class BtcGoldTest(unittest.TestCase):
    def test_packaged_models_match_research_and_are_point_in_time(self):
        data = btc_gold.load_daily()
        results = btc_gold.load_research()
        self.assertEqual(len(data), 2705)
        dashboard = btc_gold.build_dashboard_models(data)
        for expected in results["latest"]:
            rows, _ = btc_gold.snapshot(data, expected["years"])
            actual = next(row for row in rows if row["method"] == expected["method"])
            for field in ("z", "sigma", "center"):
                self.assertAlmostEqual(actual[field], expected[field], places=10)
            key = "models" if expected["method"] == "OLS" else "mean_models"
            if expected["years"] is None:
                model = dashboard["global_model" if key == "models" else "global_mean_model"]
            else:
                model = dashboard[key][expected["years"]]
            self.assertEqual(len(model), expected["n"])
            self.assertAlmostEqual(model.zscore.iloc[-1], expected["z"], places=10)
        asof = pd.Timestamp("2022-06-10")
        original, _ = btc_gold.snapshot(data, 3, asof=asof)
        mutated = data.copy()
        mutated.loc[mutated.index > asof, "ratio"] *= 100
        changed, _ = btc_gold.snapshot(mutated, 3, asof=asof)
        self.assertEqual(original, changed)

    def test_research_rejects_mismatched_daily_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "btc_gold_daily.csv").write_bytes((btc_gold.DATA_DIR / "btc_gold_daily.csv").read_bytes() + b"\n")
            (root / "btc_gold_research.json").write_bytes((btc_gold.DATA_DIR / "btc_gold_research.json").read_bytes())
            with patch.object(btc_gold, "DATA_DIR", root):
                with self.assertRaisesRegex(ValueError, "版本不一致"):
                    btc_gold.load_research()

    @patch.object(PagesManager, "uses_pages_directory", False)
    def test_detail_history_and_shared_layout(self):
        path = Path(__file__).parent / "streamlit_app.py"
        with patch("akshare.stock_zh_index_daily", fake_daily), patch("akshare.stock_zh_index_spot_sina", fake_spot):
            app = AppTest.from_file(str(path)).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            app.button(key="detail_btc_gold").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 0)
        self.assertEqual(len(app.get("plotly_chart")), 8)
        self.assertEqual(len(app.tabs), 0)
        self.assertEqual(len(app.multiselect), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe), 0)
        self.assertEqual([expander.label for expander in app.expander], ["计算口径与使用说明"])
        detail_text = " ".join(item.value for item in app.markdown)
        self.assertIn('class="detail-head"', detail_text)
        self.assertIn('class="metric-item metric-pair"', detail_text)
        self.assertIn("BTC / 黄金 固定窗口分析", detail_text)
        self.assertIn("日历年固定样本", detail_text)
        for label in ("1年", "3年", "5年", "7年", "全局", "回到最新"):
            self.assertIn(label, [button.label for button in app.button])
        self.assertNotIn("10年", [button.label for button in app.button])
        app.button(key="window_toggle_btc_gold_1").click().run(timeout=30)
        self.assertEqual(len(app.get("plotly_chart")), 6)
        self.assertNotIn("1年固定窗口", " ".join(item.value for item in app.markdown))
        app.button(key="window_toggle_btc_gold_1").click().run(timeout=30)
        self.assertEqual(len(app.get("plotly_chart")), 8)
        app.date_input[0].set_value(date(2022, 6, 10)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertIn("1年", [button.label for button in app.button])
        self.assertIn("3年", [button.label for button in app.button])
        self.assertNotIn("5年", [button.label for button in app.button])
        self.assertNotIn("7年", [button.label for button in app.button])
        self.assertIn("全局", [button.label for button in app.button])
        self.assertEqual(len(app.get("plotly_chart")), 4)
        app.date_input[0].set_value(date(2018, 2, 9)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any("不足 3 条" in item.value for item in app.info))
        app.date_input[0].set_value(date(2018, 3, 1)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 2)
        app.button(key="latest_btc_gold").click().run(timeout=30)
        self.assertEqual(app.date_input[0].value, date(2026, 10, 27))
        app.button(key="back_btc_gold").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.page, "home")
        self.assertEqual(len(app.get("plotly_chart")), 4)


if __name__ == "__main__":
    unittest.main()
