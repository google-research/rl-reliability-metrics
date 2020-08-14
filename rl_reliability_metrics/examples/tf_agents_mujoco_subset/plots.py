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
"""Step 4 of "TF-Agents Mujoco Subset" example: Make plots.

Creates plots showing the metric values of each algorithm on a per-task basis,
as well as aggregated over tasks and converted into rankings.
"""

from absl import app
from absl import flags

from rl_reliability_metrics.analysis import data_def
from rl_reliability_metrics.analysis import plotter
from rl_reliability_metrics.examples.tf_agents_mujoco_subset import params as p

FLAGS = flags.FLAGS


def make_plots():
  """Makes plots."""
  dd = data_def.DataDef(p.metric_values_dir, p.algos, p.tasks,
                        p.n_runs_per_experiment)
  my_plotter = plotter.Plotter(
      data=dd,
      pvals_dir=p.pvals_dir,
      confidence_intervals_dir=p.confidence_intervals_dir,
      n_timeframes=p.n_timeframes,
      algorithms=p.algos,
      out_dir=p.plots_dir)

  for metric in p.metrics:
    my_plotter.make_plots(metric)


def main(_):
  make_plots()


if __name__ == '__main__':
  app.run(main)
