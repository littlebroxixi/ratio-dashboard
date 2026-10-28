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
        for expected in results["latest"]:
            rows, _ = btc_gold.snapshot(data, expected["years"])
            actual = next(row for row in rows if row["method"] == expected["method"])
            for field in ("z", "sigma", "center"):
                self.assertAlmostEqual(actual[field], expected[field], places=10)
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
    def test_detail_history_and_research_filters(self):
        path = Path(__file__).parent / "streamlit_app.py"
        with patch("akshare.stock_zh_index_daily", fake_daily), patch("akshare.stock_zh_index_spot_sina", fake_spot):
            app = AppTest.from_file(str(path)).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            app.button(key="detail_btc_gold").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 0)
        self.assertEqual(len(app.get("plotly_chart")), 8)
        self.assertEqual(len(app.dataframe), 4)
        self.assertEqual(len(app.dataframe[1].value), 36)
        app.selectbox(key="fixed_year").set_value(5).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue((app.dataframe[1].value["n"] == 0).all())
        self.assertTrue(app.dataframe[1].value["mean_return"].isna().all())
        app.radio[0].set_value("收敛幅度").run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(set(app.dataframe[2].value.target_parameter) <= {0.5, 1, 1.5, 2, 2.5, 3})
        app.date_input[0].set_value(date(2022, 6, 10)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.multiselect[0].options, ["1年", "3年", "全样本"])
        self.assertEqual(len(app.get("plotly_chart")), 4)
        app.date_input[0].set_value(date(2018, 2, 9)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any("不足 3 条" in item.value for item in app.info))
        app.button(key="btc_gold_back").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.page, "home")
        self.assertEqual(len(app.get("plotly_chart")), 4)


if __name__ == "__main__":
    unittest.main()
