"""Numerical and denominator contracts for the full-agent future metrics."""
from decimal import Decimal
import json
import math
from types import SimpleNamespace
import unittest

from castfactory.training.full_agent_metrics import (
    METRIC_VERSION, forecast_metrics, metric_context, metric_decimal, metric_float,
    evaluator_preflight, normalized_group_advantages, scoring_wape, serialize_metric, stable_mean, stable_sample_std,
    stable_sum,
)


def case(truth, history=None):
    return SimpleNamespace(targets=SimpleNamespace(future_values=truth),
                           observation=SimpleNamespace(target_history=history or [10.] * 96))


class FutureMetricsTest(unittest.TestCase):
    def test_exact_wape_preserves_sub_ulp_difference_before_cost(self):
        fixture = case([0.] * 95 + [1e-12])
        better = forecast_metrics([1.] * 96, fixture)
        worse = forecast_metrics([1. + 2 ** -52] + [1.] * 95, fixture)
        self.assertEqual(better["wape"], worse["wape"])
        with metric_context():
            delta = scoring_wape(worse) - scoring_wape(better)
            reward_worse = -scoring_wape(worse) - metric_decimal(.01) * 999 / 1000
            reward_better = -scoring_wape(better) - metric_decimal(.01)
        self.assertGreater(delta, Decimal("0.0002"))
        self.assertLess(reward_worse, reward_better)
        with self.assertRaises(ValueError):
            scoring_wape({**better, "wape_decimal": "0"})

    def test_evaluator_preflight_reports_zero_denominator_and_rejects_by_default(self):
        cases = [SimpleNamespace(domain="price", observation=SimpleNamespace(sample_id="p"),
                                 targets=SimpleNamespace(future_values=[0.] * 96)),
                 SimpleNamespace(domain="solar", observation=SimpleNamespace(sample_id="s"),
                                 targets=SimpleNamespace(future_values=[1.] * 96))]
        report = evaluator_preflight(cases, reject_zero_denominator=False)
        self.assertEqual(report["zero_future_denominator"], ["p"])
        self.assertEqual(report["normalization_status"], "zero_future_denominator")
        with self.assertRaisesRegex(ValueError, "undefined.*p"):
            evaluator_preflight(cases)

    def test_evaluator_preflight_enforces_96_finite_labels(self):
        with self.assertRaises(ValueError):
            evaluator_preflight([SimpleNamespace(domain="solar", targets=SimpleNamespace(future_values=[1.] * 95))])
        with self.assertRaises(ValueError):
            evaluator_preflight([SimpleNamespace(domain="price", targets=SimpleNamespace(future_values=[float("nan")] * 96))])

    def test_signed_solar_labels_are_scored_and_reported_without_filtering(self):
        fixture = case([-.01, 1.] * 48)
        fixture.domain = "solar"
        original = list(fixture.targets.future_values)
        report = evaluator_preflight([fixture])
        self.assertEqual(report["cases"], 1)
        self.assertEqual(report["normalization_status"], "ok")
        self.assertEqual(report["negative_labels_by_domain"]["solar"],
                         {"windows": 1, "points": 48, "minimum": -.01})
        result = forecast_metrics([0.] * 96, fixture)
        self.assertEqual(result["points"], 96)
        self.assertEqual(result["wape"], 1.)
        self.assertAlmostEqual(result["abs_truth_sum"], 48.48)
        self.assertEqual(fixture.targets.future_values, original)
    def test_future_denominator_and_auxiliary_rmse(self):
        result = forecast_metrics([1., 3.] * 48, case([1., 1.] * 48))
        self.assertEqual(result["wape"], 1.)
        self.assertAlmostEqual(result["nrmse"], math.sqrt(2))
        self.assertEqual(result["abs_error_sum"], 96.)
        self.assertEqual(result["abs_truth_sum"], 96.)
        self.assertEqual(result["squared_error_sum"], 192.)
        self.assertEqual(result["nmae_hist"], 2.)
        self.assertEqual(result["metric_version"], METRIC_VERSION)
        self.assertEqual(result["normalization_status"], "ok")

    def test_signed_price_uses_absolute_truth(self):
        result = forecast_metrics([0.] * 96, case([-2., 2.] * 48))
        self.assertEqual(result["future_abs_mean"], 2.)
        self.assertEqual(result["wape"], 1.)
        self.assertEqual(result["nrmse"], 1.)

    def test_zero_future_is_undefined_even_for_perfect_forecast(self):
        for predicted in (0., 3.):
            result = forecast_metrics([predicted] * 96, case([0.] * 96))
            self.assertIsNone(result["wape"])
            self.assertIsNone(result["nrmse"])
            self.assertEqual(result["normalization_status"], "zero_future_denominator")
            self.assertEqual(result["mae"], predicted)

    def test_tiny_positive_denominator_has_no_floor(self):
        result = forecast_metrics([2e-300] * 96, case([1e-300] * 96))
        self.assertEqual(result["wape"], 1.)
        self.assertEqual(result["nrmse"], 1.)
        self.assertEqual(result["normalization_status"], "ok")
        self.assertIn("squared_error_sum", result["extended_decimal_fields"])
        self.assertGreater(metric_decimal(result["squared_error_sum"]), 0)

    def test_finite_balanced_extremes_do_not_overflow_ratio(self):
        result = forecast_metrics([-1e308, 1e308] * 48, case([1e308, -1e308] * 48))
        self.assertEqual(result["wape"], 2.)
        self.assertEqual(result["nrmse"], 2.)
        self.assertEqual(result["future_abs_mean"], 1e308)
        self.assertEqual(result["numeric_status"], "extended_decimal")
        for key in ("rmse", "mae", "abs_error_sum", "abs_truth_sum", "squared_error_sum"):
            self.assertIsInstance(result[key], str)
        json.dumps(result, allow_nan=False)

    def test_rmse_with_1e200_errors_remains_representable(self):
        result = forecast_metrics([1e200] * 96, case([0.] * 96))
        self.assertEqual(result["rmse"], 1e200)
        self.assertIsInstance(result["squared_error_sum"], str)

    def test_huge_ratio_is_serialized_not_clipped(self):
        result = forecast_metrics([1e308] * 96, case([1e-308] * 96))
        self.assertIsInstance(result["wape"], str)
        self.assertGreater(metric_decimal(result["wape"]), Decimal("1e615"))
        self.assertEqual(result["normalization_status"], "ok")
        self.assertEqual(result, forecast_metrics([1e308] * 96, case([1e-308] * 96)))
        with self.assertRaises(ValueError):
            metric_float(result["wape"])

    def test_pooled_wape_and_window_mean_are_distinct(self):
        rows = [forecast_metrics([0.] * 96, case([1.] * 96)),
                forecast_metrics([9.] * 96, case([9.] * 96))]
        self.assertEqual(stable_mean(r["wape"] for r in rows), .5)
        with metric_context():
            pooled = metric_decimal(stable_sum(r["abs_error_sum"] for r in rows)) / metric_decimal(
                stable_sum(r["abs_truth_sum"] for r in rows))
        self.assertEqual(serialize_metric(pooled), .1)

    def test_invalid_prediction_and_label_are_not_finite_extremes(self):
        for bad in (True, float("inf"), float("nan"), "1"):
            with self.assertRaises(ValueError):
                forecast_metrics([bad] * 96, case([1.] * 96))
        with self.assertRaises(ValueError):
            forecast_metrics([0.] * 96, case([float("nan")] * 96))
        missing = forecast_metrics([], case([1.] * 96))
        self.assertIsNone(missing["wape"])
        self.assertEqual(missing["normalization_status"], "invalid_forecast_shape")

    def test_historical_extremes_do_not_poison_future_metric(self):
        result = forecast_metrics([0.] * 96, case([1.] * 96, [-1e308, 1e308] * 48))
        self.assertEqual(result["wape"], 1.)
        self.assertGreater(metric_decimal(result["nmae_hist"]), 0)

    def test_legacy_diagnostic_retains_exact_manifest_scale(self):
        fixture = case([1.] * 96)
        fixture.manifest = lambda: {"history_scale": 7.25}
        result = forecast_metrics([2.] * 96, fixture)
        self.assertEqual(result["nmae_hist"], 1 / 7.25)
        self.assertEqual(result["wape"], 1.)
        fixture.manifest = lambda: {"history_scale": float("inf")}
        self.assertEqual(forecast_metrics([2.] * 96, fixture)["nmae_hist"], 2.)


class StableScalarTest(unittest.TestCase):
    def test_mean_cancellation_and_std_without_overflow(self):
        self.assertEqual(stable_mean([1e308] * 96), 1e308)
        self.assertEqual(stable_sum([1e308, 1., -1e308]), 1.)
        self.assertAlmostEqual(stable_sample_std([-1e200, 1e200]) / 1e200, math.sqrt(2))
        self.assertIsInstance(stable_sample_std([-1.7e308, 1.7e308]), str)

    def test_group_advantages_large_values_and_unchanged_epsilon(self):
        values = [-1e308, 1e308, -1e308, 1e308]
        advantages = normalized_group_advantages(values)
        for value, expected in zip(advantages, [-math.sqrt(.75), math.sqrt(.75)] * 2):
            self.assertAlmostEqual(value, expected)
        self.assertEqual(normalized_group_advantages([1e308] * 4), [0.] * 4)
        self.assertTrue(all(math.isfinite(v) for v in normalized_group_advantages(
            [1e308, -1e308, 1e-308, 1e-308])))
        small = [-1e-7, 1e-7, -1e-7, 1e-7]
        std = math.sqrt(sum(v * v for v in small) / 3)
        for actual, expected in zip(normalized_group_advantages(small), [v / (std + 1e-6) for v in small]):
            self.assertAlmostEqual(actual, expected, places=15)
        with self.assertRaises(ValueError):
            normalized_group_advantages(small, epsilon=1e-8)

    def test_decimal_roundtrip_and_explicit_range_rejection(self):
        for value in (Decimal("1e1000"), Decimal("-1e1000"), Decimal("1e-1000")):
            encoded = serialize_metric(value)
            self.assertIsInstance(encoded, str)
            self.assertEqual(metric_decimal(json.loads(json.dumps(encoded))), value)
            with self.assertRaises(ValueError):
                metric_float(encoded)
        self.assertIsInstance(serialize_metric(Decimal("3e-324")), str)
        for bad in (None, True, "NaN", "Infinity", float("nan")):
            with self.assertRaises(ValueError):
                metric_decimal(bad)


if __name__ == "__main__":
    unittest.main()
