# coding=utf-8
# Copyright 2019 The Authors of RL Reliability Metrics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for eval_metrics."""

import json
import os

from absl import flags
from absl.testing import parameterized
import gin
import numpy as np
from rl_reliability_metrics.analysis import io_utils_oss as io_utils
from rl_reliability_metrics.evaluation import eval_metrics
from rl_reliability_metrics.metrics import metrics_online

# Internal gfile dependencies
import unittest

FLAGS = flags.FLAGS


class EvalMetricsTest(parameterized.TestCase, unittest.TestCase):

  def setUp(self):
    super(EvalMetricsTest, self).setUp()

    gin.clear_config()
    gin_file = os.path.join(
        './',
        'rl_reliability_metrics/evaluation',
        'eval_metrics_test.gin')
    gin.parse_config_file(gin_file)

    # fake set of training curves to test analysis
    self.test_data_dir = os.path.join(
        './',
        'rl_reliability_metrics/evaluation/test_data')
    self.tensorboard_dirs = [  # Tensorboard data.
        os.path.join(self.test_data_dir, 'tfsummary', 'run%d' % i, 'train')
        for i in range(3)
    ]
    self.csv_paths = [  # CSV data.
        os.path.join(self.test_data_dir, 'csv', 'run%d.csv' % i)
        for i in range(2)
    ]

  def test_compute_metrics(self):
    curves = [
        np.array([[-1, 0, 1], [1., 1., 1.]]),
        np.array([[-1, 0, 1, 2], [2., 3., 4., 5.]])
    ]
    evaluator = eval_metrics.Evaluator(
        [metrics_online.StddevAcrossRuns(eval_points=[0, 1], baseline=1)])
    results = evaluator.compute_metrics(curves)
    np.testing.assert_allclose(results['StddevAcrossRuns'],
                               [1.41421356237, 2.12132034356])

  def test_window_out_of_range(self):
    curves = [np.array([[0, 1], [1, 1]])]
    evaluator = eval_metrics.Evaluator([metrics_online.StddevAcrossRuns()])
    self.assertRaises(ValueError, evaluator.compute_metrics, curves)

  def test_window_empty(self):
    curves = [np.array([[0, 2], [2, 3]])]
    evaluator = eval_metrics.Evaluator([metrics_online.StddevAcrossRuns()])
    self.assertRaises(ValueError, evaluator.compute_metrics, curves)

  def test_get_metric_params(self):
    metric_instances = [
        metrics_online.StddevAcrossRuns(eval_points=[3, 2, 1], baseline=-2),
        metrics_online.LowerCVaROnDiffs(alpha=0.77)
    ]
    metric_params = eval_metrics.get_metric_params(metric_instances)
    self.assertCountEqual(metric_params.keys(),
                          ['StddevAcrossRuns', 'LowerCVaROnDiffs'])
    self.assertEqual(
        metric_params['StddevAcrossRuns'], {
            'eval_points': [3, 2, 1],
            'baseline': -2,
            'lowpass_thresh': None,
            'window_size': None,
        })
    self.assertEqual(
        metric_params['LowerCVaROnDiffs'], {
            'target': 'diffs',
            'tail': 'lower',
            'alpha': 0.77,
            'baseline': None,
            'eval_points': None,
            'window_size': None,
            'lowpass_thresh': None,
        })

  def test_evaluate_on_tfsummary(self):
    evaluator = eval_metrics.Evaluator(
        [metrics_online.StddevWithinRuns(),
         metrics_online.StddevWithinRuns()])
    results = evaluator.evaluate(self.tensorboard_dirs)
    self.assertEqual(list(results.keys()), ['StddevWithinRuns'])
    self.assertTrue(np.greater(list(results.values()), 0.).all())

  def test_evaluate_on_csv(self):
    gin.bind_parameter('metrics_online.StddevWithinRuns.eval_points', [4001])
    gin.bind_parameter('metrics_online.StddevWithinRuns.window_size', 4001)
    evaluator = eval_metrics.Evaluator(
        [metrics_online.StddevWithinRuns(),
         metrics_online.StddevWithinRuns()],
        dependent_variable='Metrics/AverageReturn',
        timepoint_variable='Metrics/EnvironmentSteps')
    results = evaluator.evaluate(self.csv_paths)
    self.assertEqual(list(results.keys()), ['StddevWithinRuns'])
    self.assertTrue(np.greater(list(results.values()), 0.).all())

  def test_evaluate_using_environment_steps(self):
    gin.bind_parameter('metrics_online.StddevWithinRuns.eval_points', [2001])
    metric_instances = [
        metrics_online.StddevWithinRuns(),
        metrics_online.StddevWithinRuns()
    ]
    evaluator = eval_metrics.Evaluator(
        metric_instances, timepoint_variable='Metrics/EnvironmentSteps')
    results = evaluator.evaluate(self.tensorboard_dirs)
    self.assertEqual(list(results.keys()), ['StddevWithinRuns'])
    self.assertTrue(np.greater(list(results.values()), 0.).all())

  def test_write_results(self):
    # Generate some results.
    curves = [
        np.array([[-1, 0, 1], [1., 1., 1.]]),
        np.array([[-1, 0, 1, 2], [2., 3., 4., 5.]])
    ]
    metric = metrics_online.StddevAcrossRuns(eval_points=[0, 1], baseline=1)
    evaluator = eval_metrics.Evaluator([metric])
    results = evaluator.compute_metrics(curves)

    outfile_prefix = os.path.join(flags.FLAGS.test_tmpdir, 'results_')
    params_path = evaluator.write_metric_params(outfile_prefix)
    results_path = evaluator.write_results(results, outfile_prefix)

    # Test write_results.
    with open(results_path, 'r') as outfile:
      results_loaded = outfile.readline()
    results_dict = json.loads(results_loaded)
    expected = {'StddevAcrossRuns': [1.41421356237, 2.12132034356]}
    self.assertEqual(results_dict.keys(), expected.keys())
    np.testing.assert_allclose(expected['StddevAcrossRuns'],
                               results_dict['StddevAcrossRuns'])

    # Test write_metric_params.
    with open(params_path, 'r') as outfile:
      params_loaded = outfile.readline()
    expected = json.dumps({
        'StddevAcrossRuns': {
            'eval_points': [0, 1],
            'lowpass_thresh': None,
            'baseline': 1,
            'window_size': None,
        }
    })
    self.assertJsonEqual(expected, params_loaded)

  def test_permute_curves(self):
    curves = [
        np.array([[0, 1, 2], [3, 4, 5]]),
        np.array([[6, 7], [9, 10]]),
    ]
    permuted = eval_metrics.permute_curves(curves)

    # Original curves should be unchanged.
    np.testing.assert_array_equal(curves[0], np.array([[0, 1, 2], [3, 4, 5]]))
    np.testing.assert_array_equal(curves[1], np.array([[6, 7], [9, 10]]))
    # Permuted curves should have the same members.
    self.assertEqual(len(permuted), len(curves))
    for curve in permuted:
      self.assertTrue(any(np.array_equal(curve, c) for c in curves))

  def test_evaluate_with_permutations(self):
    evaluator = eval_metrics.Evaluator([metrics_online.StddevWithinRuns()])
    n_permutations = 3
    permutation_start_idx = 100
    random_seed = 50
    outfile_prefix = os.path.join(FLAGS.test_tmpdir,
                                  'robustness_results_permuted_')
    results = evaluator.evaluate_with_permutations(
        self.tensorboard_dirs, self.tensorboard_dirs, outfile_prefix,
        n_permutations, permutation_start_idx, random_seed)

    # Check length of results.
    self.assertLen(results, n_permutations)

    # Check a single result.
    one_result = list(results.values())[0]['curves1']
    self.assertEqual(list(one_result.keys()), ['StddevWithinRuns'])
    self.assertTrue(np.greater(list(one_result.values()), 0.).all())

    # Check the output files.
    results_files = io_utils.paths_glob('%s*results.json' % outfile_prefix)
    self.assertLen(results_files, 1)

    # If run again with the same seed, the results should be the same
    results_same = evaluator.evaluate_with_permutations(
        self.tensorboard_dirs, self.tensorboard_dirs, outfile_prefix,
        n_permutations, permutation_start_idx, random_seed)
    self._assert_results_same(results, results_same)

    # If run again with a different seed, the results should be different
    results_different = evaluator.evaluate_with_permutations(
        self.tensorboard_dirs, self.tensorboard_dirs, outfile_prefix,
        n_permutations, permutation_start_idx, random_seed + 1)
    self._assert_results_different(results, results_different)

  def test_resample_curves(self):
    curves = [
        np.array([[0, 1, 2], [3, 4, 5]]),
        np.array([[6, 7], [9, 10]]),
    ]
    resampled = eval_metrics.resample_curves(curves)

    # Original curves should be unchanged.
    np.testing.assert_array_equal(curves[0], np.array([[0, 1, 2], [3, 4, 5]]))
    np.testing.assert_array_equal(curves[1], np.array([[6, 7], [9, 10]]))
    # Resampled curves should have the same members.
    self.assertEqual(len(resampled), len(curves))
    for curve in resampled:
      self.assertTrue(any(np.array_equal(curve, c) for c in curves))

  def _assert_results_same(self, results_a, results_b):
    for permutation_key in results_a.keys():
      permutation_results_a = results_a[permutation_key]
      permutation_results_b = results_b[permutation_key]
      for curve in ['curves1', 'curves2']:
        self.assertEqual(permutation_results_a[curve].keys(),
                         permutation_results_b[curve].keys())
        for metric_key in permutation_results_a[curve].keys():
          np.testing.assert_array_equal(
              permutation_results_a[curve][metric_key],
              permutation_results_b[curve][metric_key])

  def _assert_results_different(self, results_a, results_b):
    for permutation_key in results_a.keys():
      permutation_results_a = results_a[permutation_key]
      permutation_results_b = results_b[permutation_key]
      for curve in ['curves1', 'curves2']:
        for metric_key in permutation_results_a[curve].keys():
          self.assertFalse(
              np.array_equal(permutation_results_a[curve][metric_key],
                             permutation_results_b[curve][metric_key]))

  @parameterized.parameters((None, ['run%i' % i for i in range(6)]),
                            (['run1', 'run3'], ['run1', 'run3']))
  def test_get_run_dirs(self, selected_runs, expected_dirs):
    tfsummary_dir = os.path.join(self.test_data_dir, 'tfsummary')
    run_dirs = eval_metrics.get_run_dirs(tfsummary_dir, 'train', selected_runs)
    run_dirs.sort()
    expected = [os.path.join(tfsummary_dir, d, 'train') for d in expected_dirs]
    self.assertEqual(run_dirs, expected)


if __name__ == '__main__':
  unittest.main()
