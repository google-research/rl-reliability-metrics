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

"""Tests for data_def objects."""

import os
from absl import flags
from rl_reliability_metrics.analysis import data_def

import unittest


class DataDefTest(unittest.TestCase):

  def test_create_datadef(self):
    results_dir = os.path.join(
        './',
        'rl_reliability_metrics/analysis/test_data')
    algorithms = ['algoA', 'algoB', 'algoC']
    tasks = ['taskX', 'taskY']

    metrics = ['IqrWithinRuns', 'MedianPerfDuringTraining', 'IqrAcrossRuns']

    dd = data_def.DataDef(
        results_dir, algorithms, tasks, n_runs_per_experiment=2)
    self.assertEqual(dd.algorithms, algorithms)
    self.assertEqual(dd.tasks, tasks)
    self.assertEqual(len(dd.results), len(dd.algorithms) * len(dd.tasks))  # pylint: disable=g-generic-assert
    self.assertCountEqual(dd.metrics, metrics)

  def test_load_empty_results(self):
    """Check for exception if loading with incorrect algorithm names."""
    results_dir = os.path.join(
        './',
        'rl_reliability_metrics/analysis/test_data')
    algorithms = ['wrong1', 'wrong2', 'wrong3']
    tasks = ['taskX', 'taskY']

    with self.assertRaises(Exception):
      data_def.DataDef(results_dir, algorithms, tasks, n_runs_per_experiment=2)


if __name__ == '__main__':
  unittest.main()
