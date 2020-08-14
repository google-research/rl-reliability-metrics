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

"""Tests for stats."""

import os
from absl import flags
from absl.testing import parameterized

import numpy as np
from rl_reliability_metrics.analysis import data_def
from rl_reliability_metrics.analysis import stats

import unittest


class StatsTest(parameterized.TestCase, unittest.TestCase):

  def setUp(self):
    super(StatsTest, self).setUp()

    results_dir = os.path.join(
        './',
        'rl_reliability_metrics/analysis/test_data')
    self.dd = data_def.DataDef(
        results_dir,
        algorithms=['algoA', 'algoB', 'algoC'],
        tasks=['taskX', 'taskY'],
        n_runs_per_experiment=2)

  @parameterized.named_parameters(
      ('2x2x1', [[[1], [2]], [[3], [4]]], 'ATR', 1),
      ('3x2x2', [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]
                ], 'ATR', 1),
      ('2x2x1x2', [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], 'ATRP', 0),
      ('2x2x5x3', [[[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]
                    ],
                    [[15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26],
                     [27, 28, 29]]],
                   [[[30, 31, 32], [33, 34, 35], [36, 37, 38], [39, 40, 41],
                     [42, 43, 44]],
                    [[45, 46, 47], [48, 49, 50], [51, 52, 53], [54, 55, 56],
                     [57, 58, 59]]]], 'ATRP', 0),
  )
  def test_resample_metric_results(self, metric_results, result_dims, algo_ind):
    stats_runner = stats.StatsRunner(
        data=self.dd, metric='IqrAcrossRuns', n_timeframes=3)
    stats_runner.result_dims = result_dims
    metric_results = np.array(metric_results)
    resampled = stats_runner._resample_metric_results(metric_results, algo_ind)

    # The resampled subarrays should be drawn from the original subarrays.
    n_task = metric_results.shape[1]
    for itask in range(n_task):
      algo_task_results = metric_results[algo_ind, itask]
      for run_value in resampled[algo_ind][itask]:
        self.assertIn(run_value, algo_task_results)

    # Resampled array should be the same for all other algorithms.
    n_metrics = metric_results.shape[0]
    other_algo_inds = list(set(range(n_metrics)) - {algo_ind})
    resampled_other_algos = resampled[other_algo_inds]
    original_other_algos = metric_results[other_algo_inds]
    np.testing.assert_array_equal(resampled_other_algos, original_other_algos)

    if metric_results.shape[2] == 1:
      # Since there is only one run, the resampled array should be the same.
      np.testing.assert_array_equal(resampled, metric_results)

  @parameterized.parameters(
      ('IqrWithinRuns', 0, [0]),
      ('IqrAcrossRuns', 1, [1]),
      ('MedianPerfDuringTraining', 2, [2, 3]),
      ('MedianPerfDuringTraining', None, [0, 1, 2, 3]),
  )
  def test_get_timeframe_points(self, metric, timeframe, expected):
    stats_runner = stats.StatsRunner(
        data=self.dd, metric=metric, n_timeframes=3)
    timeframe_points = stats_runner.get_timeframe_points(timeframe)
    np.testing.assert_array_equal(list(timeframe_points), expected)

  @parameterized.parameters(
      ('AT', True, [[2, 3], [3, 4], [1, 2], [4, 1]]),
      ('AT', False, [[3, 2], [2, 1], [4, 3], [1, 4]]),
      ('ATP', False, [[[1, 2], [1, 2]], [[3, 4], [3, 4]]]),
  )
  def test_rank_per_task(self, result_dims, bigger_is_better, expected_result):
    results_arrays = {
        'AT': [[3, 1], [2, -2], [4, 7], [0, 9]],
        'ATP': [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
    }
    results_array = np.array(results_arrays[result_dims])

    stats_runner = stats.StatsRunner(data=None, metric='IqrAcrossRuns')
    stats_runner.result_dims = result_dims
    stats_runner.bigger_is_better = bigger_is_better
    ranks = stats_runner.rank_per_task(results_array)
    np.testing.assert_array_equal(ranks, expected_result)

  def test_algo_meanrank_diff(self):
    ranks_array = np.array([[1, 2, 1, 4], [2, 4, 3, 2]])
    meanrank_diff = stats.StatsRunner._algo_meanrank_diff(ranks_array)
    self.assertEqual(meanrank_diff, 0.75)

  def test_pval(self):
    null_distribution = np.array([3, 5, 7, -2, -4, -6, 0, 0, 1, 2])
    observed_value = 5
    pval = stats.StatsRunner._get_pval(null_distribution, observed_value)
    self.assertEqual(pval, 3 / 10)


if __name__ == '__main__':
  unittest.main()
