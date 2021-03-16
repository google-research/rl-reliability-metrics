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

"""Evaluate robustness metrics for a given set of training curves or rollouts."""

import json
import os

from absl import logging
import gin
import numpy as np

from rl_reliability_metrics.analysis import io_utils_oss as io_utils
from rl_reliability_metrics.evaluation import data_loading
# Internal gfile dependencies


@gin.configurable
class Evaluator(object):
  """Class for evaluating metrics."""

  def __init__(self,
               metrics,
               dependent_variable='Metrics/AverageReturn',
               timepoint_variable=None,
               align_on_global_step=True):
    """Initialize Evaluator.

    Args:
      metrics: List of instances of robustness metrics to evaluate. See
        :metrics. Per-metric parameters (window_size, etc) may be configured
        with Gin.
      dependent_variable: Name of Tensorboard summary that should be loaded for
        analysis of robustness, e.g. 'Metrics/AverageReturn'.
      timepoint_variable: Name of Tensorboard summary that defines a "timepoint"
        (i.e. the independent variable), e.g. 'Metrics/EnvironmentSteps'. Set
        None to simply use the 'steps' value of the dependent_variable summaries
        as the timepoint variable.
      align_on_global_step: see load_input_data
    """
    self.metrics = metrics
    self.data_loader = data_loading.DataLoader(dependent_variable,
                                               timepoint_variable,
                                               align_on_global_step)

  @gin.configurable
  def evaluate(self, run_paths, outfile_prefix='/tmp/robustness_results_'):
    """Evaluate robustness metrics on a set of run directories.

    Args:
      run_paths: List of paths to directories containing Tensorboard summaries
        for all the runs of an experiment, one directory per run. Summaries must
        include a scalar or tensor summary that defines the variable to be
        analyzed (the 'dependent_variable'). Optionally they may also have a
        scalar or tensor summary that defines a "timepoint" (the
        'timepoint_variable').
      outfile_prefix: Prefix for JSON output files, where we write results and
        metric parameters.

    Returns:
      A dictionary of robustness values {metric_name: metric_value}
    """
    curves = self.data_loader.load_input_data(run_paths)

    results = self.compute_metrics(curves)
    self.write_results(results, outfile_prefix)

    return results

  def evaluate_with_permutations(
      self,
      run_paths_1,
      run_paths_2,
      outfile_prefix='/tmp/robustness_results_permuted',
      n_permutations=1000,
      permutation_start_idx=0,
      random_seed=0):
    """Evaluate robustness metrics on runs permuted across two sets.

    This method is useful for computing permutation tests to evaluate
    statistical significance on the difference in metric values between two
    sets of runs (e.g. for one algorithm vs another algorithm). In particular,
    this method is necessary to run permutation tests for across-run metrics
    (for per-run metrics, we can run permutation tests just by permuting the
    original metrics values or rankings).

    We permute the runs across the two sets and divide into two sets of the
    same size as the original two sets. We evaluate the metrics on the
    two permuted sets. This is performed n_permutations times. This provides a
    null distribution that can later be loaded to compute a p-value for a
    permutation test.

    Args:
      run_paths_1: List of paths to directories containing Tensorboard summaries
        for all the runs of an experiment, one directory per run. Summaries must
        include a scalar or tensor summary that defines the variable to be
        analyzed (the 'dependent_variable'). Optionally they may also have a
        scalar or tensor summary that defines a "timepoint" (the
        'timepoint_variable').
      run_paths_2: Another list of paths.
      outfile_prefix: Prefix for JSON output files, where we write results and
        metric parameters.
      n_permutations: Number of permutations to perform.
      permutation_start_idx: If desired, the indexing of permutations can start
        at any integer. This affects the naming of the output files.
      random_seed: Numpy random seed.

    Returns:
      A list of robustness results. Each result is a dictionary of robustness
      values {metric_name: metric_value}
    """
    np.random.seed(random_seed)

    curves_1 = self.data_loader.load_input_data(run_paths_1)
    curves_2 = self.data_loader.load_input_data(run_paths_2)
    all_curves = curves_1 + curves_2

    all_results = {}
    for i_permutation in range(permutation_start_idx,
                               permutation_start_idx + n_permutations):
      logging.info('Permutation %d...', i_permutation)
      curves_permuted = permute_curves(all_curves)
      curves_permuted_1 = curves_permuted[:len(curves_1)]
      curves_permuted_2 = curves_permuted[len(curves_1):]

      results_1 = self.compute_metrics(curves_permuted_1)
      results_2 = self.compute_metrics(curves_permuted_2)
      all_results['permutation%d' % i_permutation] = {
          'curves1': results_1,
          'curves2': results_2
      }

    permutation_end_idx = permutation_start_idx + n_permutations - 1
    outfile_prefix_extended = '%spermutations%dto%d_' % (
        outfile_prefix, permutation_start_idx, permutation_end_idx)
    self.write_results(all_results, outfile_prefix_extended)

    return all_results

  def evaluate_with_bootstraps(
      self,
      run_paths,
      outfile_prefix='/tmp/robustness_results_bootstrapped',
      n_bootstraps=1000,
      bootstrap_start_idx=0,
      random_seed=0):
    """Evaluate robustness metrics on bootstrapped runs.

    I.e. the runs are resampled with replacement.

    This method is useful for computing bootstrapped confidence intervals on
    the metric values for a single set of runs (e.g. for a single algorithm).
    In particular, this method is necessary to obtain confidence intervals for
    across-run metrics (for per-run metrics, we can obtain confidence intervals
    just by bootstrapping the original metrics values or rankings).

    We bootstrap the runs (resample with replacement) n_bootstraps times, each
    time re-computing the metrics. This provides bootstrap distributions on
    the metric values that can later be loaded to compute confidence intervals.

    Args:
      run_paths: List of paths containing data for all runs of an experiment.
        * For CSV data, this should be a list of CSV filepaths, one
        file per run. One column should define a "timepoint" (the
        'timepoint_variable'), and one column must contain the variable to be
        analyzed (the 'dependent_variable'). The name of each column should be
        in the first row.
        * For Tensorflow outputs, this should be a list of directories
        containing Tensorboard summaries for all the runs of an experiment,
        one directory per run. Summaries must include a scalar or tensor summary
        that defines the variable to be analyzed (the 'dependent_variable').
        Optionally they may also have a scalar or tensor summary that defines a
        "timepoint" (the 'timepoint_variable').
      outfile_prefix: Prefix for JSON output files, where we write results and
        metric parameters.
      n_bootstraps: Number of bootstraps to perform.
      bootstrap_start_idx: If desired, the indexing of bootstraps can start at
        any integer. This affects the naming of the output files.
      random_seed: Numpy random seed.

    Returns:
      A dict of robustness results. Each entry in the dict has the form
       {'bootstrap%%BOOTSTRAP_IDX%%': metric_result_for_this_resampling}.
      Each metric result is a dictionary of metric values
        {metric_name: metric_value}.
    """
    np.random.seed(random_seed)

    curves = self.data_loader.load_input_data(run_paths)

    all_results = {}
    for i_boot in range(bootstrap_start_idx,
                        bootstrap_start_idx + n_bootstraps):
      logging.info('Bootstrap %d...', i_boot)
      curves_resampled = resample_curves(curves)
      results_resampled = self.compute_metrics(curves_resampled)

      all_results['bootstrap%d' % i_boot] = results_resampled

    bootstrap_end_idx = bootstrap_start_idx + n_bootstraps - 1
    outfile_prefix_extended = '%sbootstraps%dto%d_' % (
        outfile_prefix, bootstrap_start_idx, bootstrap_end_idx)
    self.write_results(all_results, outfile_prefix_extended)

    return all_results

  def compute_metrics(self, curves):
    """Computes metrics on training curves."""
    results = {}
    for metric in self.metrics:
      results[metric.name] = metric(curves)
    return results

  def write_metric_params(self, outfile_prefix):
    """Write metric parameters to JSON."""

    # Load the metric params.
    metric_params = get_metric_params(self.metrics)

    # Write the metric params.
    io_utils.makedirs(os.path.dirname(outfile_prefix))
    params_path = outfile_prefix + 'metric_params.json'
    with open(params_path, 'w') as outfile:
      json.dump(metric_params, outfile, cls=_NumpyEncoder)

    logging.info('Metric params written to: %s', params_path)
    return params_path

  @staticmethod
  def write_results(results, outfile_prefix):
    """Write results to JSON."""
    io_utils.makedirs(os.path.dirname(outfile_prefix))

    results_path = outfile_prefix + 'results.json'
    with open(results_path, 'w') as outfile:
      json.dump(results, outfile, cls=_NumpyEncoder)

    logging.info('Results written to: %s', results_path)
    return results_path


def get_metric_params(metrics):
  """Gets public parameters for a list of metric instances.

  Args:
    metrics: A list of metric instances.

  Returns:
    Dictionary of metric parameters {metric_name: dict of params for metric}.
    Each entry is also a dictionary {metric_param: param_value}.
  """
  metric_params = {}
  for metric in metrics:
    metric_params[metric.name] = {
        str(attr): value
        for attr, value in vars(metric).items()
        if attr[0] != '_'
    }
  return metric_params


class _NumpyEncoder(json.JSONEncoder):

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def get_run_dirs(summary_path, data_type, selected_runs=None):
  """Get the subdirectories corresponding to each run."""
  if selected_runs:
    # only get the specified subdirectories
    run_dirs = selected_runs
  else:
    # get all the subdirectories
    run_dirs = io_utils.listdir(summary_path)
  run_dirs = [os.path.join(summary_path, d, data_type) for d in run_dirs]

  logging.info('Run directories:')
  for d in run_dirs:
    logging.info('  %s', d)

  return run_dirs


def permute_curves(curves):
  """Permute a list of curves.

  Args:
    curves: A list of curves, e.g. a 2-D Numpy array.

  Returns:
    A list of curves, with the same length as the original curves.
  """
  indices = np.random.permutation(len(curves))
  return [curves[ind] for ind in indices]


def resample_curves(curves):
  """Resample with replacement from a list of curves.

  Args:
    curves: A list of curves, e.g. a 2-D Numpy array.

  Returns:
    A list of curves, with the same length as the original curves.
  """
  n_curves = len(curves)
  indices = np.random.choice(range(n_curves), n_curves)
  return [curves[ind] for ind in indices]
