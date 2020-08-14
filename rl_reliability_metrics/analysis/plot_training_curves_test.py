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

"""Tests for plot_training_curves."""

from absl.testing import parameterized
import numpy as np
from rl_reliability_metrics.analysis import plot_training_curves as ptc

import unittest


class PlotTrainingCurvesTest(parameterized.TestCase, unittest.TestCase):

  @parameterized.parameters(
      (None, [0, 1, 2, 3], [[5, 6, 7, 8], [-1, -2, -3, -4]]),
      (1, [0.5, 1.5, 2.5, 3.5], [[5, 6, 7, 8], [-1, -2, -3, -4]]),
      (3, [1.5, 4.5], [[6, 8], [-2, -4]]),
  )
  def test_window_means(self, window_size, expected_timesteps,
                        expected_window_means):
    curves = np.array([[[0, 1, 2, 3], [5, 6, 7, 8]],
                       [[0, 1, 2, 3], [-1, -2, -3, -4]]])
    timesteps, window_means = ptc.compute_window_means(curves, window_size)
    np.testing.assert_array_equal(timesteps, expected_timesteps)
    np.testing.assert_array_equal(window_means, expected_window_means)

  def test_compute_means(self):
    window_means = [[1, 2, 3, 4], [5, 6, 7, 8]]
    means = ptc.compute_means(window_means)
    np.testing.assert_array_equal(means, [3, 4, 5, 6])

  def test_compute_medians(self):
    window_means = [[1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8]]
    means = ptc.compute_medians(window_means)
    np.testing.assert_array_equal(means, [5, 6, 7, 8])

  @parameterized.parameters(
      ([[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]),)
  def test_compute_boot_ci(self, window_means, expected):
    window_means = np.array(window_means)
    cis = ptc.compute_boot_ci(window_means)
    np.testing.assert_allclose(cis[0], expected[0])
    np.testing.assert_allclose(cis[1], expected[1])

  @parameterized.parameters(
      ([[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]),
      ([[0, -2], [1, 4]], [[0.1, -1.4], [0.80, 2.8]]),
  )
  def test_compute_percentiles(self, window_means, expected):
    window_means = np.array(window_means)
    percentiles = ptc.compute_percentiles(
        window_means, lower_thresh=10, upper_thresh=80)
    np.testing.assert_allclose(percentiles[0], expected[0])
    np.testing.assert_allclose(percentiles[1], expected[1])


if __name__ == '__main__':
  unittest.main()
