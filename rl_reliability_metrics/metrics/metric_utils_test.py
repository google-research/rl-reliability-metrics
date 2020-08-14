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

"""Tests for rl_reliability_metrics.metric_utils."""

from absl.testing import parameterized
import numpy as np
from rl_reliability_metrics.metrics import metric_utils as utils

import unittest


class MetricUtilsTest(parameterized.TestCase, unittest.TestCase):

  @parameterized.parameters((0, [1., 2., 3., 1.]), (None, 1.5))
  def testMedianAbsoluteDeviation(self, axis, expected):
    values = np.array([[1, 2, 4, 6], [2, 5, 1, 0], [0, 0, 35, -1]])
    result = utils.median_absolute_deviations(values, axis)
    np.testing.assert_allclose(result, expected)

  def testSubtractBaseline(self):
    curves = [
        np.array([[0, 1, 2], [3., 4., 5.]]),
        np.array([[-5, -7, -10], [2.4, 3.6, 0.12]])
    ]
    result = utils.subtract_baseline(curves, baseline=0.2)
    expected = [
        np.array([[0, 1, 2], [2.8, 3.8, 4.8]]),
        np.array([[-5, -7, -10], [2.2, 3.4, -0.08]])
    ]
    np.testing.assert_allclose(result, expected)

  def testDivideByBaselineSingle(self):
    curves = [
        np.array([[0, 1, 2], [3, 4., 5.]]),
        np.array([[-5, -7, -10], [2.4, 3.6, 0.12]])
    ]
    result = utils.divide_by_baseline(curves, baselines=2)
    expected = [
        np.array([[0, 1, 2], [1.5, 2, 2.5]]),
        np.array([[-5, -7, -10], [1.2, 1.8, 0.06]])
    ]
    np.testing.assert_allclose(result, expected)

  def testDivideByBaselinesList(self):
    curves = [
        np.array([[0, 1, 2], [3, 4., 5.]]),
        np.array([[-5, -7, -10], [2.4, 3.6, 0.12]])
    ]
    baselines = [1, 2]
    result = utils.divide_by_baseline(curves, baselines)
    expected = [
        np.array([[0, 1, 2], [3, 4, 5]]),
        np.array([[-5, -7, -10], [1.2, 1.8, 0.06]])
    ]
    np.testing.assert_allclose(result, expected)

  def testBandNormalization(self):
    curves = [
        np.array([[0, 1, 2], [3., 4., 5.]]),
        np.array([[-5, -7, -10], [2.4, 3.6, 0.12]])
    ]
    result = utils.band_normalization(curves, low=-0.2, high=14)
    expected = [
        np.array([[0, 1, 2], [0.22535211, 0.29577465, 0.36619718]]),
        np.array([[-5, -7, -10], [0.18309859, 0.26760563, 0.02253521]])
    ]
    np.testing.assert_allclose(result, expected)

  def testMeanNormalization(self):
    curves = [
        np.array([[0, 1, 2], [3., 4., 5.]]),
        np.array([[0, 1, 2], [-3., -4., -5.]]),
        np.array([[-5, -7, -10], [2.4, 3.6, 0.12]]),
    ]
    result = utils.mean_normalization(curves)
    expected = [
        np.array([[0, 1, 2], [0.75, 1., 1.25]]),
        np.array([[0, 1, 2], [-0.75, -1., -1.25]]),
        np.array([[-5, -7, -10], [1.176470588, 1.764705882352, 0.05882352941]])
    ]
    np.testing.assert_allclose(result, expected)

  def testMedianRolloutPerformance(self):
    rollout_sets = [
        np.array([[0, 1, 2], [3, 4, 5]]),
        np.array([[0, 1, 5], [3, 4, 5]]),
        np.array([[0, 1, 2, 3, 4], [1, 7, 2, 3, 6]]),
    ]
    expected = [4, 4, 3]
    result = utils.median_rollout_performance(rollout_sets)
    np.testing.assert_array_equal(result, expected)

  def testComputeDifferencing(self):
    curves = [
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[3, 6, 8, 9], [-1, 4, 2.3, 0]]),
    ]
    diff_curves = utils.differences(curves)
    np.testing.assert_allclose(diff_curves[0], np.array([[2, 3], [1, 1]]))
    np.testing.assert_allclose(diff_curves[1],
                               np.array([[6, 8, 9], [1.66666667, -0.85, -2.3]]))

  def testComputeDrawdown(self):
    seq = [-1, 2, 3.5, 5, 2, 1, 1.5, 3, 1.5, 6, 2]
    result = utils.compute_drawdown(seq)
    expected = [0, 0, 0, 0, 3, 4, 3.5, 2, 3.5, 0, 4]
    np.testing.assert_array_equal(result, expected)

  @parameterized.parameters(
      (None, 1, None, [[6], [-1]]),
      ([3], None, None, [[6], [-1]]),
      ([3], 1, None, [[6], [-1]]),
      ([3], 2, None, [[6], [-1]]),
      ([3], None, 0.5, [[6.22790854], [-0.82210584]]),
      ([4], 3, None, [[7], [-1]]),
      ([4, 5], 3, None, [[7, 7], [-1, 4]]),
  )
  def testAcrossRunsPreprocess(self, eval_points, window_size, lowpass_thresh,
                               expected):
    curves = [
        np.array([[1, 2, 3, 4], [4, 5, 6, 7]]),
        np.array([[3, 6, 8, 9], [-1, 4, 2.3, 0]]),
    ]
    if eval_points is not None:
      eval_points = np.array(eval_points)
    result = utils.across_runs_preprocess(curves, eval_points, window_size,
                                          lowpass_thresh)
    np.testing.assert_allclose(result, expected)

  def testAcrossRunsPreprocessInvalidInputs(self):
    curves = [
        np.array([[1, 2, 3, 4], [4, 5, 6, 7]]),
        np.array([[3, 6, 8, 9], [-1, 4, 2.3, 0]]),
    ]
    self.assertRaises(
        ValueError,
        utils.across_runs_preprocess,
        curves,
        eval_points=np.array([4]),
        window_size=2,
        lowpass_thresh=None)

  @parameterized.parameters(
      ('lower', 0.3, 0.75),
      ('upper', 0.1, 5.5),
  )
  def testComputeCVaR(self, tail, alpha, expected):
    seq = np.array([-1, 2, 3.5, 5, 2, 1, 1.5, 3, 1.5, 6, 2])
    result = utils.compute_cvar(seq, tail, alpha)
    np.testing.assert_array_equal(result, expected)

  def testFindNearest(self):
    array = np.array([3, 1, 200, -4, 200])
    targets = np.array([2, 1, -5, 150])
    nearest_values, distances = utils.find_nearest(array, targets)
    np.testing.assert_array_equal(nearest_values, [3, 1, -4, 200])
    np.testing.assert_array_equal(distances, [1, 0, 1, 50])

  def testGetNearestWithinWindow(self):
    curves = [
        np.array([[1, 2, 3, 4], [4, 5, 6, 7]]),
        np.array([[3, 6, 8, 9], [-1, 4, 2.3, 0]]),
    ]
    eval_points = np.array([2, 3, 6])
    window_size = 5
    nearest_vals = utils.get_nearest_within_window(curves, eval_points,
                                                   window_size)
    np.testing.assert_array_equal(nearest_vals,
                                  np.array([[5, 6, 7], [-1, -1, 4]]))

  def testCurveRange(self):
    curves = [
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[3, 6, 8, 9], [-1, 4, 2.3, 0]]),
    ]
    result = utils.curve_range(curves)
    expected = [1.9, 4.745]
    np.testing.assert_allclose(result, expected)

  @parameterized.parameters(
      (1, [np.array([[0, 2], [4, 6]]),
           np.array([[0], [10]])]),
      ('drop', [np.array([4, 6]), np.array([10])]))
  def testApplyFnToCurves(self, trim, expected):
    curves = [np.array([[0, 1, 2], [1, 2, 3]]), np.array([[0, 1], [4, 5]])]

    def fn(curve):
      return curve[1:] * 2

    result = utils.apply_fn_to_curves(curves, fn, trim)
    for result_single, expected_single in zip(result, expected):
      np.testing.assert_array_equal(result_single, expected_single)

  def testApplyWindowFn(self):
    curves = [np.array([[0, 1, 2], [1, 2, 3]]), np.array([[0, 1], [4, 5]])]
    eval_points = [0, 1]
    curve_points = utils.apply_window_fn(curves, eval_points, np.mean)
    np.testing.assert_array_equal(curve_points, np.array([[1, 2], [4, 5]]))


if __name__ == '__main__':
  unittest.main()
