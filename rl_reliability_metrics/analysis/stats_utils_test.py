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

"""Tests for plotter object."""

from absl.testing import parameterized

import numpy as np
from rl_reliability_metrics.analysis import stats_utils

import unittest


class StatsUtilsTest(parameterized.TestCase, unittest.TestCase):

  @parameterized.parameters(
      (2, 0.05, 0.05),
      (4, 0.05, 0.00833333),
  )
  def test_bonferroni_correction(self, n_algorithms, alpha, expected):
    pthresh_corrected = stats_utils.multiple_comparisons_correction(
        n_algorithms, alpha, 'bonferroni')
    self.assertAlmostEqual(pthresh_corrected, expected)

  def test_benjamini_yekutieli_correction(self):
    pthresh_corrected = stats_utils.multiple_comparisons_correction(
        n_algorithms=4, alpha=0.05, method='benjamini-yekutieli')
    expected = [
        0.05 * 1 / 6,
        0.05 * 2 / 6 / (1 + 1 / 2),
        0.05 * 3 / 6 / (1 + 1 / 2 + 1 / 3),
        0.05 * 4 / 6 / (1 + 1 / 2 + 1 / 3 + 1 / 4),
        0.05 * 5 / 6 / (1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5),
        0.05 / (1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5 + 1 / 6),
    ]
    np.testing.assert_allclose(pthresh_corrected, expected)


if __name__ == '__main__':
  unittest.main()
