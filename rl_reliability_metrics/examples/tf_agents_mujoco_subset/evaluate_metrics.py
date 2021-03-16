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
"""Step 1 of "TF-Agents Mujoco Subset" example: Evaluate the metrics."""

import os

from absl import app
from absl import flags

import gin
from rl_reliability_metrics.evaluation import eval_metrics
from rl_reliability_metrics.examples.tf_agents_mujoco_subset import params as p

FLAGS = flags.FLAGS

flags.DEFINE_enum('resampling', 'none', ['none', 'permute', 'bootstrap'],
                  'The type of resampling to apply to the runs.')


def evaluate_metrics():
  """Evaluates metrics specified in the gin config."""
  # Parse gin config.
  gin.parse_config_files_and_bindings([p.gin_file], [])

  for algo in p.algos:
    for task in p.tasks:
      run_paths = _get_run_paths(algo, task)

      # Evaluate metrics.
      outfile_prefix = os.path.join(p.metric_values_dir, algo, task) + '/'
      evaluator = eval_metrics.Evaluator(metrics=gin.REQUIRED)
      evaluator.write_metric_params(outfile_prefix)
      evaluator.evaluate(run_paths=run_paths, outfile_prefix=outfile_prefix)


def evaluate_metrics_on_permuted_runs():
  """Evaluates metrics on permuted runs, for across-run metrics only."""
  gin_bindings = [
      ('eval_metrics.Evaluator.metrics = '
       '[@IqrAcrossRuns/singleton(), @LowerCVaROnAcross/singleton()]')
  ]
  n_permutations_per_worker = int(p.n_random_samples / p.n_worker)

  # Parse gin config.
  gin.parse_config_files_and_bindings([p.gin_file], gin_bindings)

  for algo1 in p.algos:
    for algo2 in p.algos:
      for task in p.tasks:
        for i_worker in range(p.n_worker):
          # Get the subdirectories corresponding to each run.
          run_paths_1 = _get_run_paths(algo1, task)
          run_paths_2 = _get_run_paths(algo2, task)

          # Evaluate the metrics.
          outfile_prefix = os.path.join(p.metric_values_dir_permuted, '%s_%s' %
                                        (algo1, algo2), task) + '/'
          evaluator = eval_metrics.Evaluator(metrics=gin.REQUIRED)
          evaluator.write_metric_params(outfile_prefix)
          evaluator.evaluate_with_permutations(
              run_paths_1=run_paths_1,
              run_paths_2=run_paths_2,
              outfile_prefix=outfile_prefix,
              n_permutations=n_permutations_per_worker,
              permutation_start_idx=(n_permutations_per_worker * i_worker),
              random_seed=i_worker)


def evaluate_metrics_on_bootstrapped_runs():
  """Evaluates metrics on bootstrapped runs, for across-run metrics only."""
  gin_bindings = [
      'eval_metrics.Evaluator.metrics = [@IqrAcrossRuns/singleton(), '
      '@LowerCVaROnAcross/singleton()]'
  ]
  n_bootstraps_per_worker = int(p.n_random_samples / p.n_worker)

  # Parse gin config.
  gin.parse_config_files_and_bindings([p.gin_file], gin_bindings)

  for algo in p.algos:
    for task in p.tasks:
      for i_worker in range(p.n_worker):
        # Get the subdirectories corresponding to each run.
        run_paths = _get_run_paths(algo, task)

        # Evaluate results.
        outfile_prefix = os.path.join(p.metric_values_dir_bootstrapped, algo,
                                      task) + '/'
        evaluator = eval_metrics.Evaluator(metrics=gin.REQUIRED)
        evaluator.write_metric_params(outfile_prefix)
        evaluator.evaluate_with_bootstraps(
            run_paths=run_paths,
            outfile_prefix=outfile_prefix,
            n_bootstraps=n_bootstraps_per_worker,
            bootstrap_start_idx=(n_bootstraps_per_worker * i_worker),
            random_seed=i_worker)


def _get_run_paths(algo, task):
  """Get paths to experiment runs data."""
  if p.data_source == 'csv':
    # Get the paths to the CSV files corresponding to each run.
    run_paths = [
        os.path.join(p.data_dir, 'csv_data',
                     '%s_%s_%s_train.csv' % (algo, task, run))
        for run in p.runs
    ]
  elif p.data_source == 'tensorboard':
    # Get the subdirectories containing Tensorboard summaries corresponding
    # to each run.
    summary_path = os.path.join(p.data_dir, algo, task)
    run_paths = eval_metrics.get_run_dirs(summary_path, 'train', p.runs)
  else:
    raise ValueError('Invalid value for params.data_source: %s' %
                     p.data_source)
  return run_paths


def main(_):
  if FLAGS.resampling == 'none':
    evaluate_metrics()
  elif FLAGS.resampling == 'permute':
    evaluate_metrics_on_permuted_runs()
  elif FLAGS.resampling == 'bootstrap':
    evaluate_metrics_on_bootstrapped_runs()


if __name__ == '__main__':
  app.run(main)
