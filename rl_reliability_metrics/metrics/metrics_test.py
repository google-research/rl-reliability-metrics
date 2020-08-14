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

"""General tests for metrics."""

import os
from absl import flags

from absl.testing import parameterized
import gin
import numpy as np
from rl_reliability_metrics.metrics import metrics_base
from rl_reliability_metrics.metrics import metrics_online

import unittest


ALL_METRICS = metrics_base.all_metrics()


class MetricsTest(parameterized.TestCase, unittest.TestCase):

  def setUp(self):
    super(MetricsTest, self).setUp()

    gin_file = os.path.join(
        './',
        'rl_reliability_metrics/metrics',
        'metrics_test.gin')
    gin.parse_config_file(gin_file)

  def testGetPublicSubclasses(self):

    class BaseClass(metrics_base.Metric):
      pass

    class _PrivateSubClass(BaseClass):
      pass

    class PublicSubSubClass(_PrivateSubClass):
      pass

    class PublicSubClass(BaseClass):
      pass

    class _PrivateSubSubClass(PublicSubClass):  # pylint: disable=unused-variable
      pass

    result = BaseClass.public_subclasses()
    expected = {PublicSubSubClass, PublicSubClass}
    self.assertEqual(set(result), expected)

  @parameterized.parameters(
      ([np.array([[], []])], [0]),
      ([np.array([range(2), [1, 2]]),
        np.array([[0], [3]])], [-2, -1]),
      ([np.array([range(2), [1, 2]]),
        np.array([[0], [3]])], [2]))
  def testErrorOnEvalPointsOutOfBounds(self, curves, eval_points):
    metric = metrics_online.StddevAcrossRuns(lowpass_thresh=None,
                                             eval_points=eval_points)
    self.assertRaises(ValueError, metric, curves)

  def testErrorOnEmptyList(self):
    curves = []
    for metric in ALL_METRICS:
      self.assertRaises(ValueError, metric(), curves)


if __name__ == '__main__':
  unittest.main()
