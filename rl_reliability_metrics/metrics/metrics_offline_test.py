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

"""Tests for offline metrics."""

from absl.testing import parameterized
import numpy as np
from rl_reliability_metrics.metrics import metrics_offline

import unittest


class MetricsOfflineTest(parameterized.TestCase, unittest.TestCase):

  def testCorrectIqrAcrossRollouts(self):
    rollout_sets = [
        np.array([range(5), [-1, 2, 5, 7, 3]]),
        np.array([range(5), [6, 5, 1, 34, 2]])
    ]
    expected = [3 / 3, 4 / 5]
    metric = metrics_offline.IqrAcrossRollouts(baseline='median_perf')
    result = metric(rollout_sets)
    np.testing.assert_array_equal(result, expected)

  def testCorrectMadAcrossRollouts(self):
    rollout_sets = [
        np.array([range(5), [-1, 2, 5, 7, 3]]),
        np.array([range(5), [6, 5, 1, 34, 2]])
    ]
    expected = [2 / 3, 3 / 3]
    metric = metrics_offline.MadAcrossRollouts(baseline=3)
    result = metric(rollout_sets)
    np.testing.assert_array_equal(result, expected)

  def testCorrectStddevAcrossRollouts(self):
    rollout_sets = [
        np.array([range(5), [-1, 2, 5, 7, 3]]),
        np.array([range(5), [6, 5, 1, 34, 2]])
    ]
    expected = [3.03315017762062 * 2, 13.794926603646717 * 2]
    metric = metrics_offline.StddevAcrossRollouts(baseline=0.5)
    result = metric(rollout_sets)
    np.testing.assert_allclose(result, expected)

  def testCorrectLowerCVaRAcrossRollouts(self):
    rollout_sets = [
        np.array([range(5), [-1, 2, 5, 7, 3]]),
        np.array([range(5), [6, 5, 1, 34, 2]]),
    ]
    metric = metrics_offline.LowerCVaRAcrossRollouts(alpha=0.49, baseline=0.5)
    result = metric(rollout_sets)
    expected = [0.5 / 0.5, 1.5 / 0.5]
    np.testing.assert_allclose(result, expected)

  def testCorrectUpperCVaRAcrossRollouts(self):
    rollout_sets = [
        np.array([range(5), [-1, 2, 5, 7, 3]]),
        np.array([range(5), [6, 5, 1, 34, 2]]),
    ]
    metric = metrics_offline.UpperCVaRAcrossRollouts(alpha=0.49, baseline=0.7)
    result = metric(rollout_sets)
    expected = [6 / 0.7, 20 / 0.7]
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      ([np.array([range(4), [1, 2, 3, 4]])], None, [2.5]),
      ([np.array([range(5), [-1, -2, -3, -4, -5]]),
        np.array([range(4), [1, 3, 2, 1.2]])], None, [-3, 1.6]),
      ([np.array([range(3), [3, 6, 9]]),
        np.array([range(2), [7.5, 7.7]])], None, [6, 7.6]),
      ([np.array([range(3), [3, 6, 9]]),
        np.array([range(2), [7.5, 7.7]])], 0.8, [6 / 0.8, 7.6 / 0.8]),
      ([np.array([range(2), [7.5, 7.7]])], (0.8, 10.3), [0.71578947368]),
      ([np.array([range(3), [7.5, 7.7, 8.0]])], None, [7.7]),
  )
  def testCorrectMedianPerf(self, rollout_sets, baseline, expected):
    metric = metrics_offline.MedianPerfAcrossRollouts(baseline)
    result = metric(rollout_sets)
    np.testing.assert_allclose(result, expected)

  def testMetricProperties_BiggerIsBetter(self):
    for metric in metrics_offline.all_offline_metrics():
      if metric.__name__ in [
          'LowerCVaRAcrossRollouts', 'UpperCVaRAcrossRollouts',
          'MedianPerfAcrossRollouts'
      ]:
        self.assertTrue(metric.bigger_is_better)
      elif metric.__name__ in ['MadAcrossRollouts', 'IqrAcrossRollouts', 'StddevAcrossRollouts']:
        self.assertFalse(metric.bigger_is_better)
      else:
        raise ValueError('Metric %s not accounted for.' % metric.__name__)

  def testMetricProperties_ResultDimensions(self):
    for metric in metrics_offline.all_offline_metrics():
      self.assertEqual(metric.result_dimensions, 'ATR')

  def testRegistry(self):
    registry = metrics_offline.REGISTRY
    self.assertEqual(registry['MedianPerfAcrossRollouts'],
                     metrics_offline.MedianPerfAcrossRollouts)
    self.assertEqual(registry['MadAcrossRollouts'],
                     metrics_offline.MadAcrossRollouts)


if __name__ == '__main__':
  unittest.main()
