import unittest

import numpy as np
import pandas as pd

from analytics import (
    build_pair_models,
    consensus_snapshot,
    fixed_mean_signal,
    fixed_ols_signal,
    slice_asof,
)


def sample_base(length=2100):
    index = pd.bdate_range("2015-01-01", periods=length)
    x = np.arange(length, dtype=float)
    ratio = 1.10 - 0.00003 * x + 0.018 * np.sin(x / 37.0) + 0.006 * np.cos(x / 11.0)
    return pd.DataFrame(
        {"ratio": ratio, "price_a": ratio * 5000.0, "price_b": 5000.0},
        index=index,
    )


class FixedOlsTests(unittest.TestCase):
    def test_asof_slice_uses_previous_trading_day_and_excludes_future(self):
        base = sample_base()
        sliced, actual = slice_asof(base, "2022-06-11")

        self.assertEqual(actual, pd.Timestamp("2022-06-10"))
        self.assertEqual(sliced.index[-1], pd.Timestamp("2022-06-10"))
        self.assertTrue((sliced.index <= actual).all())
        self.assertLess(len(sliced), len(base))

    def test_latest_matches_direct_ols(self):
        base = sample_base()
        model = fixed_ols_signal(base, 3)
        latest = model.iloc[-1]
        window = 3 * 252
        history = base["ratio"].iloc[-window:].to_numpy()
        x = np.arange(window, dtype=float)
        design = np.column_stack((np.ones(window), x))
        beta, *_ = np.linalg.lstsq(design, history, rcond=None)
        fitted = design @ beta
        expected_center = fitted[-1]
        expected_sigma = np.sqrt(np.sum((history - fitted) ** 2) / (window - 2))
        expected_z = (history[-1] - expected_center) / expected_sigma

        self.assertAlmostEqual(latest["center"], expected_center, places=10)
        self.assertAlmostEqual(latest["residual_sigma"], expected_sigma, places=10)
        self.assertAlmostEqual(latest["zscore"], expected_z, places=10)

    def test_uses_exact_latest_fixed_sample(self):
        base = sample_base()
        model = fixed_ols_signal(base, 5)
        self.assertEqual(len(model), 5 * 252)
        self.assertEqual(model.index[0], base.index[-5 * 252])
        self.assertEqual(model.index[-1], base.index[-1])
        self.assertAlmostEqual(model["residual_sigma"].std(), 0.0, places=12)

    def test_consensus_counts_same_direction_confirmations(self):
        index = pd.DatetimeIndex([pd.Timestamp("2026-01-01")])
        models = {
            3: pd.DataFrame({"zscore": [-1.8]}, index=index),
            5: pd.DataFrame({"zscore": [-2.2]}, index=index),
            7: pd.DataFrame({"zscore": [-2.7]}, index=index),
        }
        result = consensus_snapshot(models)
        self.assertEqual(result.direction, -1)
        self.assertEqual(result.confirmation_count, 2)
        self.assertEqual(result.strong_count, 1)
        self.assertEqual(result.label, "双周期确认")

    def test_fixed_mean_matches_direct_sample_statistics(self):
        base = sample_base()
        model = fixed_mean_signal(base, 3)
        history = base["ratio"].iloc[-3 * 252 :]
        latest = model.iloc[-1]

        self.assertAlmostEqual(latest["center"], history.mean(), places=12)
        self.assertAlmostEqual(latest["residual_sigma"], history.std(ddof=1), places=12)
        self.assertAlmostEqual(
            latest["zscore"],
            (history.iloc[-1] - history.mean()) / history.std(ddof=1),
            places=12,
        )
        self.assertAlmostEqual(model["center"].std(), 0.0, places=12)
        self.assertAlmostEqual(model["residual_sigma"].std(), 0.0, places=12)

    def test_pair_models_include_matching_mean_windows(self):
        df = sample_base().rename(columns={"price_a": "a", "price_b": "b"})[["a", "b"]]
        result = build_pair_models(df, "a", "b")

        self.assertEqual(set(result["models"]), {1, 3, 5, 7})
        self.assertEqual(set(result["mean_models"]), {1, 3, 5, 7})
        self.assertTrue(result["models"][5].index.equals(result["mean_models"][5].index))
        self.assertEqual(len(result["global_model"]), len(df))
        self.assertTrue(result["global_model"].index.equals(result["global_mean_model"].index))

    def test_long_history_unlocks_10_15_and_20_year_windows(self):
        df = sample_base(5200).rename(columns={"price_a": "a", "price_b": "b"})[["a", "b"]]
        result = build_pair_models(df, "a", "b")

        self.assertEqual(set(result["models"]), {1, 3, 5, 7, 10, 15, 20})
        self.assertEqual(len(result["models"][20]), 20 * 252)
        self.assertEqual(len(result["global_model"]), 5200)


if __name__ == "__main__":
    unittest.main()
