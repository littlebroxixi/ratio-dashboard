import unittest

import numpy as np
import pandas as pd

from analytics import consensus_snapshot, rolling_ols_signal


def sample_base(length=2100):
    index = pd.bdate_range("2015-01-01", periods=length)
    x = np.arange(length, dtype=float)
    ratio = 1.10 - 0.00003 * x + 0.018 * np.sin(x / 37.0) + 0.006 * np.cos(x / 11.0)
    return pd.DataFrame(
        {"ratio": ratio, "price_a": ratio * 5000.0, "price_b": 5000.0},
        index=index,
    )


class RollingOlsTests(unittest.TestCase):
    def test_latest_matches_direct_ols(self):
        base = sample_base()
        model = rolling_ols_signal(base, 3)
        latest = model.iloc[-1]
        window = 3 * 252
        history = base["ratio"].iloc[-window - 1 : -1].to_numpy()
        x = np.arange(window, dtype=float)
        design = np.column_stack((np.ones(window), x))
        beta, *_ = np.linalg.lstsq(design, history, rcond=None)
        fitted = design @ beta
        expected_center = beta[0] + beta[1] * window
        expected_sigma = np.sqrt(np.sum((history - fitted) ** 2) / (window - 2))
        expected_z = (base["ratio"].iloc[-1] - expected_center) / expected_sigma

        self.assertAlmostEqual(latest["center"], expected_center, places=10)
        self.assertAlmostEqual(latest["residual_sigma"], expected_sigma, places=10)
        self.assertAlmostEqual(latest["zscore"], expected_z, places=10)

    def test_current_observation_does_not_move_its_own_anchor(self):
        base = sample_base()
        changed = base.copy()
        changed.iloc[-1, changed.columns.get_loc("ratio")] += 0.25
        original_latest = rolling_ols_signal(base, 3).iloc[-1]
        changed_latest = rolling_ols_signal(changed, 3).iloc[-1]

        self.assertAlmostEqual(original_latest["center"], changed_latest["center"], places=12)
        self.assertAlmostEqual(
            original_latest["residual_sigma"], changed_latest["residual_sigma"], places=12
        )
        self.assertNotAlmostEqual(original_latest["zscore"], changed_latest["zscore"], places=3)

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


if __name__ == "__main__":
    unittest.main()
