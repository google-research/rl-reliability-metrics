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

# Lint as: python3
"""Step 2 of "TF-Agents Mujoco Subset" example: Permutation tests.

Performs permutation tests to compare metric values for different algorithms, to
determine whether differences between algorithms are statistically significant.
"""

from absl import app
from absl import flags

from rl_reliability_metrics.analysis import data_def
from rl_reliability_metrics.analysis import stats
from rl_reliability_metrics.examples.tf_agents_mujoco_subset import params as p

FLAGS = flags.FLAGS


def permutation_tests():
  """Evaluates permutation tests."""
  # Evaluate metrics that only have one value per training run.
  for metric in p.metrics_no_timeframes:
    for algo1 in p.algos:
      for algo2 in p.algos:
        data = data_def.DataDef(
            p.metric_values_dir, p.algos, p.tasks, p.n_runs_per_experiment)
        stats_runner = stats.StatsRunner(data, metric, None,
                                         p.n_random_samples, p.pvals_dir,
                                         p.metric_values_dir_permuted)
        stats_runner.compare_algorithms(algo1, algo2, timeframe=None)

  # Evaluate metrics computed at different points along each training run.
  for metric in p.metrics_with_timeframes:
    for algo1 in p.algos:
      for algo2 in p.algos:
        for timeframe in p.timeframes:
          data = data_def.DataDef(
              p.metric_values_dir, p.algos, p.tasks, p.n_runs_per_experiment)
          stats_runner = stats.StatsRunner(data, metric, p.n_timeframes,
                                           p.n_random_samples, p.pvals_dir,
                                           p.metric_values_dir_permuted)
          stats_runner.compare_algorithms(algo1, algo2, timeframe)


def main(_):
  permutation_tests()


if __name__ == '__main__':
  app.run(main)
