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

"""Class for storing results of robustness metrics."""

import copy
import json
import os

from absl import logging
from rl_reliability_metrics.analysis import io_utils_oss as io_utils
# Internal gfile dependencies


class DataDef(object):
  """Object that loads and holds a set of metric results."""

  def __init__(self,
               results_dir,
               algorithms,
               tasks,
               n_runs_per_experiment,
               dataset=None):
    self.results_dir = results_dir
    self.algorithms = algorithms
    self.tasks = tasks
    self.n_runs_per_experiment = n_runs_per_experiment
    self.dataset = dataset

    # Load results.
    self.n_tasks = len(self.tasks)
    self.results = self._load_results()
    self._check_results()

    # Get the metric names.
    self.metrics, self.metrics_robustness = self._get_metrics(self.results)

    # Get metric params for every metric.
    metric_params_single = self._get_metric_params(self.algorithms[0],
                                                   self.tasks[0])
    self.metric_params = metric_params_single

  def _load_results(self):
    """Load all results."""
    logging.info('Loading all results...')
    results = {}
    for algo in self.algorithms:
      for task in self.tasks:
        results_path = '%s/%s/%s/results.json' % (self.results_dir, algo, task)
        with open(results_path) as results_file:
          results['%s.%s' % (algo, task)] = json.load(results_file)
    return results

  def _check_results(self):
    """Check that loaded results are not empty."""
    if not self.results:
      raise ValueError(
          'Did not load any results from results_dir (may be empty or '
          'incorrectly specified): %s' % self.results_dir)

  @staticmethod
  def _get_metrics(results):
    """Get list of all metrics, and list of robustness metrics."""
    one_result = list(results.values())[0]
    metrics = sorted(one_result.keys())

    metrics_robustness = copy.deepcopy(metrics)
    for metric in metrics_robustness:
      if metric in ['MedianPerfDuringTraining', 'MedianPerfAcrossRollouts']:
        metrics_robustness.remove(metric)

    return metrics, metrics_robustness

  def _get_metric_params(self, algo, task):
    """Get metric params for a single (algo, task) combination."""
    params_path = '%s/%s/%s/metric_params.json' % (self.results_dir, algo, task)
    with open(params_path) as results_file:
      metric_params = json.load(results_file)
    return metric_params


class DataDefPermuted(DataDef):
  """Object to hold metric results on permuted runs.

  We only load for a single pair of algorithms at a time.
  """

  def __init__(self, results_dir, algorithms, tasks, n_runs_per_experiment,
               n_permutation):
    if len(algorithms) != 2:
      raise ValueError('len(algorithms) should be 2, not %d.' % len(algorithms))
    self.n_permutation = n_permutation

    super(DataDefPermuted, self).__init__(results_dir, algorithms, tasks,
                                          n_runs_per_experiment)

  def _load_results(self):
    """Load all results for the particular pair of algorithms."""
    logging.info('Loading all results...')
    results_basedir = os.path.join(self.results_dir,
                                   '%s_%s' % tuple(self.algorithms))
    results = {}
    for task in self.tasks:
      # Load a set of results
      results_filepaths = io_utils.paths_glob('%s/%s/permutations*_results.json'
                                              % (results_basedir, task))
      for filepath in results_filepaths:
        logging.info('Loading %s', filepath)
        with open(filepath) as results_file:
          results_set = json.load(results_file)

          # Parse the results for each permutation and split
          for permutation_key, permutation_results in results_set.items():
            seed = int(permutation_key.split('permutation')[1])
            for split in [1, 2]:
              result_key = '%s.seed%d.split%d' % (task, seed, split)
              results[result_key] = permutation_results['curves%d' % split]

    logging.info('Results loaded:')
    logging.info(results)
    return results

  def _get_metric_params(self, algo, task):
    """Get metric params for a single (algo, task) combination."""
    params_path = '%s/%s_%s/%s/metric_params.json' % (self.results_dir, algo,
                                                      algo, task)
    with open(params_path) as results_file:
      metric_params = json.load(results_file)
    return metric_params


class DataDefBootstrapped(DataDef):
  """Data definition for robustness results on bootstrapped runs.

  We only load for a single algorithm at a time.
  """

  def __init__(self, results_dir, algorithm, tasks, n_runs_per_experiment,
               n_bootstrap):
    self.n_bootstrap = n_bootstrap
    self.algorithm = algorithm

    super(DataDefBootstrapped, self).__init__(results_dir, [algorithm], tasks,
                                              n_runs_per_experiment)

  def _load_results(self):
    """Load all results for the algorithm."""
    logging.info('Loading all results...')
    results_basedir = os.path.join(self.results_dir, self.algorithm)
    results = {}
    for task in self.tasks:
      # Load a set of results
      results_filepaths = io_utils.paths_glob('%s/%s/bootstraps*_results.json'
                                              % (results_basedir, task))
      for filepath in results_filepaths:
        logging.info('Loading %s', filepath)
        with open(filepath) as results_file:
          results_set = json.load(results_file)

          # Parse the results for each bootstrap and split
          for bootstrap_key, bootstrap_results in results_set.items():
            seed = int(bootstrap_key.split('bootstrap')[1])
            result_key = '%s.seed%d' % (task, seed)
            results[result_key] = bootstrap_results

    logging.info('Results loaded:')
    logging.info(results)
    return results
