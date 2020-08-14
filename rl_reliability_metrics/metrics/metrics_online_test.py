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

"""Tests for online metrics."""

from absl.testing import parameterized
import numpy as np
from rl_reliability_metrics.metrics import metrics_online

import unittest


class MetricsOnlineTest(parameterized.TestCase, unittest.TestCase):

  @parameterized.parameters(
      ([0, 1], None, None, [1.41421356237, 2.12132034356]),
      ([0, 1], None, 1, [1.41421356237, 2.12132034356]),
      ([0, 1], 0.5, 0.5, [2.9954294688643497, 4.564952035367936]),
      ([0, 1], None, 'curve_range', [1.414213562 / 1.425, 2.121320343 / 1.425]),
  )
  def testCorrectStddevAcrossRuns(self, timepoints, lowpass_thresh, baseline,
                                  expected):
    curves = [
        np.array([[-1, 0, 1], [1., 1., 1.]]),
        np.array([[-1, 0, 1, 2], [2., 3., 4., 5.]])
    ]
    metric = metrics_online.StddevAcrossRuns(
        lowpass_thresh=lowpass_thresh,
        eval_points=timepoints,
        baseline=baseline)
    result = metric(curves)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      ([0, 1], None, None, [1, 1.5]),
      ([0, 1], None, 2, [0.5, 0.75]),
  )
  def testCorrectIqrAcrossRuns(self, timepoints, lowpass_thresh, baseline,
                               expected):
    curves = [
        np.array([[-1, 0, 1], [1., 1., 1.]]),
        np.array([[-1, 0, 1, 2], [2., 3., 4., 5.]])
    ]
    metric = metrics_online.IqrAcrossRuns(
        lowpass_thresh=lowpass_thresh,
        eval_points=timepoints,
        baseline=baseline)
    result = metric(curves)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      ([0, 1], None, None, [1.75, 0.]),
      ([0, 1], None, 0.5, [3.5, 0.]),
  )
  def testCorrectMadAcrossRuns(self, timepoints, lowpass_thresh, baseline,
                               expected):
    curves = [
        np.array([[-1, 0, 1, 2], [1., 1., 1., 1.]]),
        np.array([[-1, 0, 1, 2], [2., 3., 4., 5.]]),
        np.array([[-1, 0, 1, 2], [3., 4.5, 1., 23.]]),
        np.array([[-1, 0, 1, 2], [0., -1., 1, 0.]]),
    ]
    metric = metrics_online.MadAcrossRuns(
        lowpass_thresh=lowpass_thresh,
        eval_points=timepoints,
        baseline=baseline)
    result = metric(curves)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      ([9], 3, None, [[0.], [0.], [0.35355339059]]),
      ([9], 3, 0.5, [[0.], [0.], [0.7071067812]]),
      (None, None, None, [[0.], [0.], [0.28867513411]]),
      (None, None, 2, [[0.], [0.], [0.1443375671]]),
      (None, None, 'curve_range', [[np.nan], [0.], [-1.9245008946666669]]),
  )
  def testCorrectStddevWithinRuns(self, timepoints, window_size, baseline,
                                  expected):
    curves = [
        np.array([[5, 7, 9], [1, 1, 1]]),
        np.array([[5, 7, 9, 11], [2, 3, 4, 5]]),
        np.array([[5, 7, 9, 10], [5, 4, 2, 1]])
    ]
    metric = metrics_online.StddevWithinRuns(window_size, timepoints, baseline)
    result = metric(curves)
    self.assertEqual(metric.name, 'StddevWithinRuns')
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      ([9], 3, None, [[0.], [0.], [0.25]]),
      ([9], 3, 0.5, [[0.], [0.], [.5]]),
      (None, None, None, [[0.], [0.], [0.25]]),
      (None, None, 2, [[0.], [0.], [0.125]]),
  )
  def testCorrectIqrWithinRuns(self, timepoints, window_size, baseline,
                               expected):
    curves = [
        np.array([[5, 7, 9], [1, 1, 1]]),
        np.array([[5, 7, 9, 11], [2, 3, 4, 5]]),
        np.array([[5, 7, 9, 10], [5, 4, 2, 1]]),
    ]
    metric = metrics_online.IqrWithinRuns(window_size, timepoints, baseline)
    result = metric(curves)
    self.assertEqual(metric.name, 'IqrWithinRuns')
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(([9], 3, None, [[0.], [0.], [0.25]]),
                            ([9], 3, 0.5, [[0.], [0.], [.5]]),
                            (None, None, None, [[0.], [0.], [0.]]),
                            (None, None, 2, [[0.], [0.], [0.]]))
  def testCorrectMadWithinRuns(self, timepoints, window_size, baseline,
                               expected):
    curves = [
        np.array([[5, 7, 9], [1, 1, 1]]),
        np.array([[5, 7, 9, 11], [2, 3, 4, 5]]),
        np.array([[5, 7, 9, 10], [5, 4, 2, 1]]),
    ]
    metric = metrics_online.MadWithinRuns(window_size, timepoints, baseline)
    result = metric(curves)
    self.assertEqual(metric.name, 'MadWithinRuns')
    np.testing.assert_allclose(result, expected)

  def testHighFreqEnergyWithinRuns(self):
    t = np.arange(0, 50, 0.1)
    sine1 = np.sin(t - np.pi / 4)  # frequency 1/(2*pi) = 0.159
    sine2 = 2 * np.sin(2 * t)  # frequency 2/(2*pi) = 0.318
    curves = [np.array([t, sine1 + sine2])]

    thresh_0 = metrics_online.HighFreqEnergyWithinRuns(thresh=0)(curves)[0]
    thresh_158 = metrics_online.HighFreqEnergyWithinRuns(
        thresh=0.158)(curves)[0]
    thresh_16 = metrics_online.HighFreqEnergyWithinRuns(thresh=0.16)(curves)[0]
    thresh_20 = metrics_online.HighFreqEnergyWithinRuns(thresh=0.2)(curves)[0]
    thresh_32 = metrics_online.HighFreqEnergyWithinRuns(thresh=0.32)(curves)[0]

    self.assertGreater(thresh_0, thresh_158)
    self.assertGreater(thresh_158, thresh_20)
    self.assertGreater(thresh_20, thresh_32)
    self.assertEqual(thresh_0, 1)
    self.assertTrue(np.allclose(thresh_158, 1, rtol=8e-3))
    self.assertFalse(np.allclose(thresh_16, 1, rtol=8e-3))

  @parameterized.parameters(
      ([np.array([range(4), [1, 2, 3, 4]])], False, None, [0]),
      ([np.array([range(4), [1, 2, 3, 4]])], True, 0.5, [0]), ([
          np.array([range(5), [-1, -2, -3, -4, -5]]),
          np.array([range(4), [1, 3, 2, 1.2]])
      ], False, None, [4, 1.8]),
      ([np.array([range(5), [5, 5, 3, 6, 5]])], False, None, [2]),
      ([np.array([range(6), [100, 150, 90, 120, 80, 200]])], False, None, [70]),
      ([np.array([range(6), [100, 150, 90, 120, 80, 200]])
       ], True, 10, [0.61764706]))
  def testCorrectMaxDrawdown(self, curves, mean_normalize, baseline, expected):
    metric = metrics_online.MaxDrawdown(baseline, mean_normalize)
    result = metric(curves)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      (metrics_online.LowerCVaROnRaw, 0.51, None, [1.5, -0.2]),
      (metrics_online.LowerCVaROnRaw, 0.49, 0.1, [1.5 / 0.1, -0.2 / 0.1]),
      (metrics_online.LowerCVaROnRaw, 0.49, 'curve_range',
       [1.5 / 2.85, -0.2 / 0.595]),
      (metrics_online.LowerCVaROnDiffs, 0.51, None, [0.8333333, -0.4]),
      (metrics_online.LowerCVaROnDiffs, 0.49, 1, [0.5, -0.6]),
      (metrics_online.LowerCVaROnDiffs, 0.51, 2, [0.8333333 / 2, -0.4 / 2]),
      (metrics_online.LowerCVaROnDiffs, 0.49, 2, [0.5 / 2, -0.6 / 2]),
      (metrics_online.LowerCVaROnDrawdown, 0.51, None, [0, 0]),
      (metrics_online.LowerCVaROnDrawdown, 0.49, 2, [0, 0]),
      (metrics_online.UpperCVaROnDrawdown, 0.25, None, [0, 0.8]),
      (metrics_online.UpperCVaROnDrawdown, 0.49, 2, [0, 0.5 / 2]),
      (metrics_online.LowerCVaROnAcross, 0.51, None, [0.1, -0.5]),
      (metrics_online.LowerCVaROnAcross, 0.49, 2, [0.05, -0.25]),
  )
  def testCorrectCVaR(self, cvar_fn, alpha, baseline, expected):
    curves = [
        np.array([[1, 2, 4, 5], [1, 2, 3, 4]]),
        np.array([range(4), [0.3, 0.1, -0.5, 1]])
    ]
    metric = cvar_fn(alpha, baseline)
    result = metric(curves)
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      ([np.array([[101, 201, 301, 401], [1, 2, 3, 4]])], [101, 401
                                                         ], 1, None, [[1, 4]]),
      ([
          np.array([[1, 1001, 2001, 3001, 4001], [-1, -2, -3, -4, -5]]),
          np.array([[1, 1001, 2001, 3001], [1, 3, 2, 1.2]])
      ], [1001, 2001], 1001, None, [[-1.5, -2.5], [2, 2.5]]),
      ([np.array([range(3), [3, 6, 9]]),
        np.array([range(2), [7.5, 7.7]])], None, None, None, [[4.5], [7.6]]),
      ([np.array([range(3), [3, 6, 9]]),
        np.array([range(2), [7.5, 7.7]])], None, None, 0.8, [[4.5 / 0.8],
                                                             [7.6 / 0.8]]),
      ([np.array([range(2), [7.5, 7.7]])], None, None,
       (0.8, 10.3), [[0.71578947368]]),
      ([np.array([range(3), [7.5, 7.7, 1.0]])], None, None, None, [[7.5]]),
  )
  def testCorrectMedianPerf(self, curves, timepoints, window_size, baseline,
                            expected):
    metric = metrics_online.MedianPerfDuringTraining(window_size, timepoints,
                                                     baseline)
    result = metric(curves)
    np.testing.assert_allclose(result, expected)

  def testMetricProperties_BiggerIsBetter(self):
    for metric in metrics_online.all_online_metrics():
      if metric.__name__ in [
          'LowerCVaROnAcross', 'LowerCVaROnDiffs', 'UpperCVaROnAcross',
          'UpperCVaROnDiffs', 'LowerCVaROnRaw', 'UpperCVaROnRaw',
          'MedianPerfDuringTraining'
      ]:
        self.assertTrue(metric.bigger_is_better)
      elif metric.__name__ in [
          'IqrAcrossRuns', 'MadAcrossRuns', 'StddevAcrossRuns', 'IqrWithinRuns',
          'MadWithinRuns', 'StddevWithinRuns', 'HighFreqEnergyWithinRuns',
          'MaxDrawdown', 'LowerCVaROnDrawdown', 'UpperCVaROnDrawdown'
      ]:
        self.assertFalse(metric.bigger_is_better)
      else:
        raise ValueError('Metric %s not accounted for.' % metric.__name__)

  def testMetricProperties_ResultDimensions(self):
    for metric in metrics_online.all_online_metrics():
      if metric.__name__ in [
          'IqrAcrossRuns', 'MadAcrossRuns', 'StddevAcrossRuns',
          'LowerCVaROnAcross', 'UpperCVaROnAcross'
      ]:
        self.assertEqual(metric.result_dimensions, 'ATP')
      elif metric.__name__ in [
          'IqrWithinRuns', 'MadWithinRuns', 'StddevWithinRuns',
          'MedianPerfDuringTraining'
      ]:
        self.assertEqual(metric.result_dimensions, 'ATRP')
      elif metric.__name__ in [
          'LowerCVaROnDiffs', 'LowerCVaROnDrawdown', 'LowerCVaROnRaw',
          'UpperCVaROnDiffs', 'UpperCVaROnDrawdown', 'UpperCVaROnRaw',
          'MaxDrawdown', 'HighFreqEnergyWithinRuns'
      ]:
        self.assertEqual(metric.result_dimensions, 'ATR')
      else:
        raise ValueError('Metric %s not accounted for.' % metric.__name__)

  def testRegistry(self):
    registry = metrics_online.REGISTRY
    self.assertEqual(registry['MedianPerfDuringTraining'],
                     metrics_online.MedianPerfDuringTraining)
    self.assertEqual(registry['IqrWithinRuns'], metrics_online.IqrWithinRuns)
    self.assertEqual(registry['LowerCVaROnDiffs'],
                     metrics_online.LowerCVaROnDiffs)


if __name__ == '__main__':
  unittest.main()
