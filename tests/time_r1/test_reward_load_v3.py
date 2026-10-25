import importlib.util
import json
import math
from pathlib import Path
import unittest

if __import__("importlib.util", fromlist=["find_spec"]).find_spec("numpy") is None:
    raise unittest.SkipTest("install elecload[reasoning] or elecload[train]")
import numpy as np

from castfactory.time_r1 import reward_load_v3 as reward
from castfactory.training.chat_template_compat import PROMPT_OWNED_RESPONSE_PREFIX_KEY


def response(values):
    return '<think>Use the observed load.</think><answer>' + json.dumps({"values": values}) + '</answer>'


class LoadRewardTests(unittest.TestCase):
    def test_bounded_profile_preserves_raw_metrics_and_uplift_units(self):
        info = {'model_input': {'target_history': {'values': [130.] * 96}},
                'reward_references': {'no_exo_ref_wape': .3, 'no_exo_ref_nrmse': .3,
                                     'no_exo_ref_metric_contract': reward.METRIC_CONTRACT}}
        raw = reward.compute_score('load', response([110.] * 24), [100.] * 24, info)
        normalized = reward.compute_score('load', response([110.] * 24), [100.] * 24, info,
                                          outcome_profile='bounded-v1')
        for key in raw:
            if key not in ('score', 'outcome'):
                self.assertEqual(raw[key], normalized[key], key)
        self.assertAlmostEqual(normalized['outcome'], -2.)
        self.assertEqual(normalized['raw_outcome'], raw['outcome'])
        self.assertEqual(normalized['outcome_valid'], normalized['outcome'])
        self.assertEqual(normalized['normalization_active'], 1.)
        self.assertNotIn('normalization_active', raw)
        self.assertEqual(reward.compute_score('load', response([110.] * 24), [100.] * 24, info,
                                             outcome_profile='raw-v1'), raw)
        self.assertEqual(normalized['score'], sum(normalized[k] for k in reward.DEFAULT_REWARD_COMPONENT_KEYS))

    def test_bounded_profile_orders_extremes_above_invalid(self):
        scores = []
        for error in (0, .03, .05, .1, .2, .5, 1., 4.):
            result = reward.compute_score('load', response([100 * (1 + error)] * 24), [100.] * 24,
                                          outcome_profile='bounded-v1', uplift_weight=0)
            scores.append(result['score'])
        self.assertEqual(scores[0], 0)
        self.assertTrue(all(a > b for a, b in zip(scores, scores[1:])))
        for factor in (-1e12, 0, 1e12):
            result = reward.compute_score('load', response([factor] * 24), [100.] * 24,
                                          outcome_profile='bounded-v1', uplift_weight=0)
            self.assertTrue(math.isfinite(result['score']))
            self.assertGreater(result['score'], reward.HARD_INVALID)
        bad = reward.compute_score('load', 'malformed', [100.] * 24, outcome_profile='bounded-v1')
        self.assertEqual(bad['score'], -6.)
        self.assertEqual(bad['format'], -6.)
        self.assertTrue(math.isnan(bad['raw_outcome']))
        self.assertTrue(math.isnan(bad['outcome_valid']))
        self.assertTrue(math.isnan(bad['wape']))
        for kwargs in ({'outcome_profile': 'unknown'}, {'half_saturation_loss': 0},
                       {'half_saturation_loss': True}, {'half_saturation_loss': float('inf')},
                       {'outcome_bound': 5}, {'outcome_bound': -1}):
            with self.assertRaises(ValueError):
                reward.compute_score('load', 'malformed', [100.] * 24, **kwargs)

    def test_bounded_transform_extreme_scale_and_mixed_group_signal(self):
        for scale in (1e-300, .1, 1e308):
            for loss in (0, 1e-300, .1, 4., 1e308):
                value = reward._outcome_loss(loss, half_saturation_loss=scale, outcome_bound=4.)
                self.assertTrue(math.isfinite(value))
                self.assertTrue(0 <= value <= 4.)
        losses = np.array([.08, .1, .12])
        raw = np.append(-losses, -6)
        new = np.append([-reward._outcome_loss(x, half_saturation_loss=.1, outcome_bound=4)
                         for x in losses], -6)
        whiten = lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-6)
        a, b = whiten(raw), whiten(new)
        self.assertGreater(b[0] - b[2], 10 * (a[0] - a[2]))
        self.assertLess(b[3], 0)
        # Numerical shaping preserves physical-unit invariance, independently
        # of the pre-existing auxiliary penalty floors.
        for factor in (1e-12, 1e12):
            metrics = reward._normalized_errors([110 * factor] * 24, [100 * factor] * 24)
            self.assertAlmostEqual(reward._outcome_loss(reward._metric_loss(*metrics),
                                   half_saturation_loss=.1, outcome_bound=4), 2.)

    def test_dynamic_horizons_metrics_and_reward(self):
        for h in (24, 96):
            with self.subTest(h=h):
                target = [100., 200., 300., 400.] * (h // 4)
                pred = [110., 220., 330., 440.] * (h // 4)
                result = reward.compute_score("load", response(pred), target, uplift_weight=0)
                expected_rmse = math.sqrt((100+400+900+1600)/4)/250
                self.assertAlmostEqual(result["wape"], .1)
                self.assertAlmostEqual(result["nrmse"], expected_rmse)
                self.assertAlmostEqual(result["metric_loss"], .9*.1+.1*expected_rmse)
                self.assertAlmostEqual(result["score"], -result["metric_loss"])
                self.assertNotIn("nmae", result); self.assertNotIn("nmse", result)

    def test_json_contract_rejects_bad_or_legacy_answers(self):
        invalid = ['<think>x</think><answer>{"values":[1,2],"extra":0}</answer>',
                   '<think>x</think><answer>{"values":[1,2],"values":[1,2]}</answer>',
                   '<think>x</think><answer>{"values":[true,2]}</answer>',
                   '<think>x</think><answer>{"values":[NaN,2]}</answer>',
                   '<think>x</think><answer>{"values":[1e999,2]}</answer>',
                   '<think>x</think><answer>```\n1 | 1\n2 | 2\n```</answer>',
                   response([1]), response([1,2,3]), response([1,2])+'extra',
                   '<think></think><answer>{"values":[1,2]}</answer>']
        invalid.append('<think>\n \t</think><answer>{"values":[1,2]}</answer>')
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(reward.compute_score("load", text, [1,2])["score"], -6)

    def test_native_prompt_owned_think(self):
        raw = response([1,2])[len('<think>'):]
        self.assertEqual(reward.compute_score("load", raw, [1,2])["score"], -6)
        scored = reward.compute_score("load", raw, [1,2], {PROMPT_OWNED_RESPONSE_PREFIX_KEY:'<think>'})
        self.assertEqual(scored['outcome'], 0)

    def test_verl_label_envelopes_and_invalid_metrics(self):
        for label in ([1., 2.], [[1.], [2.]]):
            result = reward.compute_score('load', response([1., 2.]), {'label': label})
            self.assertEqual(result['forecast_valid'], 1)
            self.assertEqual(result['wape'], 0)
        bad = reward.compute_score('load', 'malformed', {'label': [1., 2.]})
        self.assertEqual(bad['score'], -6)
        self.assertEqual(bad['forecast_valid'], 0)
        self.assertTrue(math.isnan(bad['wape']))
        self.assertTrue(math.isnan(bad['nrmse']))
        with self.assertRaises(ValueError):
            reward.extract_ground_truth_values({'label': [[1., 2.], [3., 4.]]})

    def test_target_scale_unit_invariance_and_signed_labels(self):
        target = [-1., 1., 3., 5.]; prediction = [-2., 2., 4., 6.]
        actual = reward._normalized_errors(prediction,target)
        self.assertAlmostEqual(actual[0], .4); self.assertAlmostEqual(actual[1], .4)
        for factor in (1e-12,1e12):
            np.testing.assert_allclose(actual,reward._normalized_errors(np.array(prediction)*factor,np.array(target)*factor))
        with self.assertRaisesRegex(ValueError, 'zero future'):
            reward.compute_score('load',response([0,0]),[0,0])

    def test_hgb_and_history_use_new_metrics(self):
        refs={'no_exo_ref_wape':.2,'no_exo_ref_nrmse':.4,
              'no_exo_ref_metric_contract':reward.METRIC_CONTRACT}
        info={'model_input':{'target_history':{'values':[1000]*96}},'reward_references':refs}
        result=reward.compute_score('load',response([100]*24),[100]*24,info)
        self.assertAlmostEqual(result['no_exo_baseline_loss'],.22)
        self.assertEqual(result['baseline_source'],2)
        self.assertGreater(result['uplift_bonus'],0)
        self.assertEqual(result['history_baseline_available'],1)
        missing_contract = {'reward_references': {'no_exo_ref_wape': .2, 'no_exo_ref_nrmse': .4}}
        self.assertEqual(reward.compute_score('load',response([100]*24),[100]*24,missing_contract)['no_exo_baseline_available'],0)
        info['reward_references']={'no_exo_ref_nmae':.01,'no_exo_ref_nmse':.01}
        result=reward.compute_score('load',response([100]*24),[100]*24,info)
        self.assertEqual(result['no_exo_baseline_available'],0)
        self.assertEqual(result['baseline_source'],1)

    def test_weight_cap_config_and_file_loading(self):
        self.assertEqual(reward._metric_loss(20,30,wape_weight=.9,metric_cap=4),4)
        for kwargs in ({'wape_weight':2},{'metric_cap':float('nan')},{'uplift_weight':-1}):
            with self.assertRaises(ValueError):reward.compute_score('load',response([1,2]),[1,2],**kwargs)
        spec=importlib.util.spec_from_file_location('load_reward_file_test',Path(reward.__file__))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.compute_score('load',response([1,2]),json.dumps([1,2]))['outcome'],0)


if __name__ == '__main__':
    unittest.main()
