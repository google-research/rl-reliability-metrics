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

"""Library for running analyses/statistics on robustness metric results.

Two types of analyses can be performed:
  (1) bootstrap runs for an algorithm to obtain confidence intervals
  (2) permute runs across pairs of algorithms to compare

For online metrics, we can compute these for different timeframes along the
training runs (e.g. beginning/middle/end).
"""
from absl import logging
import numpy as np
from rl_reliability_metrics.analysis import data_def
from rl_reliability_metrics.analysis import io_utils_oss as io_utils
from rl_reliability_metrics.metrics import metrics_offline
from rl_reliability_metrics.metrics import metrics_online
import scipy.stats

# Internal gfile dependencies


class StatsRunner(object):
  """Computes statistics on robustness metric results, for a single metric."""

  def __init__(self,
               data,
               metric,
               n_timeframes=3,
               n_random_samples=1000,
               outfile_dir=None,
               resampled_results_dir=None):
    """Initialize StatsRunner object.

    Args:
      data: DataDef object containing metric results.
      metric: Which metric to evaluate.
      n_timeframes: Number of timeframes that we are splitting each curve into.
      n_random_samples: Number of random samples (number of permutations /
        number of bootstraps).
      outfile_dir: Path to directory in which to write outputs, if desired.
      resampled_results_dir: Path to directory containing metric values
        evaluated on permuted or bootstrapped runs (only necessary for
        across-run metrics).
    """
    self.data_def = data
    self.metric = metric
    self.n_timeframes = n_timeframes
    self.n_random_samples = n_random_samples
    self.outfile_dir = outfile_dir
    self.resampled_results_dir = resampled_results_dir

    # Get metric-specific information
    self.result_dims, self.bigger_is_better = self._get_metric_attributes()

  def compare_algorithms(self, algo1, algo2, timeframe):
    """Compute statistical significance on difference between two algorithms.

    Args:
      algo1: First algorithm for comparison
      algo2: Second algorithm for comparison.
      timeframe: Index of the timeframe that we are evaluating.

    Returns:
      p-value for the permutation test comparing the two algorithms.
    """
    timeframe_points = self.get_timeframe_points(timeframe)

    # Load the metric results.
    metric_results = self.load_metric_results(self.data_def.algorithms,
                                              timeframe_points)

    # Compute the ranks of each algorithm, per task
    metric_ranks_all = self.rank_per_task(metric_results)
    algo1_ind = self.data_def.algorithms.index(algo1)
    algo2_ind = self.data_def.algorithms.index(algo2)
    metric_ranks_algo1_algo2 = metric_ranks_all[[algo1_ind, algo2_ind]]

    # Compute the actual difference in mean rank between algorithms.
    actual_diff = self._algo_meanrank_diff(metric_ranks_algo1_algo2)

    # Get the null distribution of differences in mean rank, through permutation
    if self.result_dims == 'ATP':
      # For across-run metrics, we load already-computed metrics values that
      # were evaluated on permuted runs. For each permutation, we rank and
      # compute the difference in mean rank to obtain a null distribution.
      perm_diffs = self._load_metrics_on_permuted_and_diff_rank(
          algo1, algo2, timeframe_points)
    else:
      # Directly permute the metric ranks to obtain a null distribution.
      perm_diffs = self._permute_ranks_and_diff(metric_ranks_algo1_algo2)

    # Get p-value from the permutation distribution.
    pval = self._get_pval(perm_diffs, actual_diff)

    # Write result.
    if self.outfile_dir:
      self._write_pval_result(pval, algo1, algo2, timeframe)
    logging.info('P-value: %g', pval)

    return pval

  def bootstrap_confidence_interval(self, algo, timeframe, alpha=0.05):
    """Compute confidence interval for an algorithm, by bootstrapping on runs.

    Args:
      algo: Algorithm to evaluate.
      timeframe: Index of the timeframe that we are evaluating.
      alpha: Threshold for confidence interval. Confidence level = 1 - alpha.

    Returns:
      (lower bound, upper bound) on the confidence interval.
    """
    timeframe_points = self.get_timeframe_points(timeframe)

    # Get the bootstrap distribution on the mean rank.
    if self.result_dims == 'ATP':
      # For across-run metrics, load the metrics evaluated on bootstrapped runs,
      # then rank to get the bootstrap distribution.
      algo_mean_ranks = self._bootstrap_distribution_loaded(
          algo, timeframe_points)
    else:
      # Bootstrap the runs on the metric results directly.
      metric_results = self.load_metric_results(self.data_def.algorithms,
                                                timeframe_points)
      algo_ind = self.data_def.algorithms.index(algo)
      algo_mean_ranks = self._bootstrap_distribution_directly(
          metric_results, algo_ind)

    # Compute the confidence interval
    ci_lower = np.percentile(algo_mean_ranks, 100 * alpha / 2)
    ci_upper = np.percentile(algo_mean_ranks, 100 * (1 - (alpha / 2)))

    # Write result.
    if self.outfile_dir:
      self._write_confidence_interval_result(ci_lower, ci_upper, algo,
                                             timeframe)
    logging.info('Confidence interval: (%s, %s)', ci_lower, ci_upper)

    return ci_lower, ci_upper

  def _get_metric_attributes(self):
    registry = {}
    registry.update(metrics_offline.REGISTRY)
    registry.update(metrics_online.REGISTRY)
    metric_cls = registry[self.metric]
    return metric_cls.result_dimensions, metric_cls.bigger_is_better

  def _bootstrap_distribution_loaded(self, algo, timeframe_points):
    """Get distribution on mean ranks, by loading bootstrapped metric values.

    For across-run metrics (result_dims=='ATP'), we cannot directly bootstrap
    the metric rankings, because the metrics need to be re-evaluated for each
    bootstrap resampling.

    Here we load the metric results that were evaluated on bootstrapped runs
    (resampled with replacement within each algo/task combination). Then, for
    each resampling, we compute rankings and mean ranking, in order to obtain a
    bootstrap distribution on the mean ranking.

    Args:
      algo: The algorithm to be evaluated.
      timeframe_points: List of indices to load, along the "eval points"
        dimension.

    Returns:
      Bootstrap distribution on mean rank of the algorithm.
      A 1-D Numpy array with length self.n_random_samples
    """
    assert self.result_dims == 'ATP'  # across-run metrics

    # Load the metrics values, evaluated on resampled runs.
    data_bootstrapped = data_def.DataDefBootstrapped(
        self.resampled_results_dir,
        algorithm=algo,
        tasks=self.data_def.tasks,
        n_runs_per_experiment=self.data_def.n_runs_per_experiment,
        n_bootstrap=self.n_random_samples)

    # Compute the bootstrap distribution of mean rankings.
    mean_ranks = np.empty(self.n_random_samples)
    for i_boot in range(self.n_random_samples):
      # Load the bootstrapped metric values.
      bootstrapped_metric_values = np.empty(self.data_def.n_tasks)
      for i_task, task in enumerate(self.data_def.tasks):
        all_timepoints = data_bootstrapped.results['%s.seed%d' %
                                                   (task, i_boot)][self.metric]
        timeframe_values = np.array(all_timepoints)[timeframe_points]
        bootstrapped_metric_values[i_task] = np.median(timeframe_values)

      # Combine bootstrapped + non-bootstrapped metric values
      other_algos = set(self.data_def.algorithms) - {algo}
      nonbootstrapped_metric_values = self.load_metric_results(
          other_algos, timeframe_points)
      all_metric_values = np.concatenate(
          (nonbootstrapped_metric_values, [bootstrapped_metric_values]), axis=0)

      # Rank the metric values
      metric_ranks_all = self.rank_per_task(all_metric_values)

      # Compute the mean rank, for the algorithm of interest.
      algo_ranks = metric_ranks_all[-1, :]
      mean_ranks[i_boot] = np.mean(algo_ranks)

    return mean_ranks

  def _bootstrap_distribution_directly(self, metric_results, algo_ind):
    """Resample runs on the metric results, to get a distribution of mean ranks.

    Args:
      metric_results: Array containing metric values for all algorithms.
      algo_ind: The index of the algorithm that we wish to resample for.

    Returns:
      Bootstrap distribution on mean rank of the algorithm.
      A 1-D Numpy array with length self.n_random_samples
    """
    # Bootstrap the metric results and get distribution of mean rank.
    algo_mean_ranks = np.empty(self.n_random_samples)
    for iboot in range(self.n_random_samples):
      # Resample the per-run metric results, only for algorithm of interest.
      metric_results_resampled = self._resample_metric_results(
          metric_results, algo_ind)

      # Compute the ranks of each algorithm, per task
      metric_ranks = self.rank_per_task(metric_results_resampled)

      # Get the mean rank for the algorithm of interest.
      algo_mean_ranks[iboot] = np.mean(metric_ranks[algo_ind])

    return algo_mean_ranks

  def _resample_metric_results(self, metric_results, algo_ind):
    """Resample runs with replacement, only for the algorithm of interest.

    Args:
      metric_results: Array containing metric values for all algorithms.
      algo_ind: The index of the algorithm that we wish to resample for.

    Returns:
      Array that is the same as metric_results, except that the metric values
      have been resampled with replacement (within each task) for the algorithm
      of interest.
    """
    assert self.result_dims[:3] == 'ATR'
    n_task = metric_results.shape[1]
    n_runs = metric_results.shape[2]

    # Resample the runs with replacement within each task, for the specified
    # algorithm only.
    resampled = metric_results.copy()
    for task_ind in range(n_task):
      # Resample.
      resampling_inds = np.random.choice(
          range(n_runs), size=n_runs, replace=True)
      algo_task_results = metric_results[algo_ind, task_ind]
      algo_task_results_resampled = algo_task_results[resampling_inds]

      # Place back into full array of metric results.
      resampled[algo_ind, task_ind] = algo_task_results_resampled

    return resampled

  def get_timeframe_points(self, timeframe):
    """Determine which timepoints are in this timeframe.

    The eval points will be divided into timeframes of equal length. If the
    total length of the time series is not equally divisible, the remainder will
    be assigned to the last timeframe.

    Args:
      timeframe: Which timeframe to get. To get the indices corresponding to
        *all* evaluation points, set to None.

    Returns:
      A list of indices corresponding to the points within the desired
      timeframe.
    """
    # Some metrics have no eval points (just one value per run).
    if 'P' not in self.result_dims:
      return None

    # Get the evaluation points for this metric.
    eval_points = np.array(
        self.data_def.metric_params[self.metric]['eval_points'])

    # Return indices corresponding to all evaluation points, if indicated.
    if timeframe is None:
      return range(len(eval_points))

    # Get the length of each timeframe.
    timeframe_len = (eval_points[-1] - eval_points[0]) / self.n_timeframes

    # Get timeframe start.
    if timeframe == 0:
      start_idx = 0
    else:
      start_time = timeframe * timeframe_len
      start_idx = np.where(eval_points >= start_time)[0][0]

    # Get timeframe end.
    if timeframe == (self.n_timeframes - 1):
      end_idx = len(eval_points)
    else:
      end_time = (timeframe + 1) * timeframe_len
      end_idx = np.where(eval_points < end_time)[0][-1] + 1

    return range(start_idx, end_idx)

  def load_metric_results(self,
                          algos,
                          timeframe_points,
                          collapse_on_timepoints=True):
    """Load all results for the metric, for the specified timeframe.

    Args:
      algos: List of strings specifying which algorithms to load.
      timeframe_points: List of indices to load. May be None if the metric has
        no "eval points" dimension (i.e. "P" not in self.result_dims)
      collapse_on_timepoints: If True, we collapse across all timepoints within
        the timeframe.

    Returns:
      Numpy array containing the metric results for the specified metric and
      algorithms. Dimensions as specified by self.result_dims.
    """
    n_algo = len(algos)
    n_task = len(self.data_def.tasks)
    n_run = self.data_def.n_runs_per_experiment
    timeframe_len = len(timeframe_points) if timeframe_points else None

    # Initialize the array
    if self.result_dims == 'ATP':  # (algo, task, evalpoint)
      metric_results = np.empty((n_algo, n_task, timeframe_len))
    elif self.result_dims == 'ATRP':  # (algo, task, run, evalpoint)
      metric_results = np.empty((n_algo, n_task, n_run, timeframe_len))
    elif self.result_dims == 'ATR':  # (algo, task, run)
      metric_results = np.empty((n_algo, n_task, n_run))
    else:
      raise ValueError('Cannot currently process dimensions: %s' %
                       self.result_dims)

    # Load results for each algo and task
    for i_algo, algo in enumerate(algos):
      for i_task, task in enumerate(self.data_def.tasks):
        algo_task_results = self.data_def.results['%s.%s' %
                                                  (algo, task)][self.metric]
        algo_task_results = np.array(algo_task_results)

        # take only the timeframe points, if needed
        if self.result_dims == 'ATRP':
          algo_task_results = algo_task_results[:, timeframe_points]
        elif self.result_dims == 'ATP':
          algo_task_results = algo_task_results[timeframe_points]

        metric_results[i_algo][i_task] = algo_task_results

    # Compute median across eval points within the timeframe
    if collapse_on_timepoints and ('P' in self.result_dims):
      metric_results = np.median(metric_results, -1)

    return metric_results

  def rank_per_task(self, results_array):
    """Rank all results, on a per-task basis.

    Ranks start at 1, which indicates the best algorithm.

    Args:
      results_array: Array with dimensions self.result_dims

    Returns:
      Array with the rank for each result, evaluated separately for each task.
      Dimensions same as results_array.
    """
    task_dim = self.result_dims.find('T')
    n_task = results_array.shape[task_dim]
    assert task_dim == 1  # We make this assumption below.

    ranks = np.empty(results_array.shape)
    for i_task in range(n_task):
      task_results = results_array[:, i_task]
      task_ranks = scipy.stats.rankdata(task_results)
      if self.bigger_is_better:
        # bigger metric values are better (and should have smaller rank).
        task_ranks = task_ranks.size + 1 - task_ranks
      task_ranks = np.reshape(task_ranks, task_results.shape)
      ranks[:, i_task] = task_ranks

    return ranks

  @staticmethod
  def _algo_meanrank_diff(ranks_array):
    """Compute the difference in mean rank, for two algorithms.

    Args:
      ranks_array: Array with the ranks for each algorithm, for each task.
        ranks_array[0] are the ranks for the first algorithm. ranks_array[1] are
        the ranks for the second algorithm.

    Returns:
      Float: Mean rank of second algorithm minus mean rank of first algorithm.
    """
    assert ranks_array.shape[0] == 2

    # For each algo, compute mean rank across tasks (and runs, if needed).
    algo1_meanrank = np.mean(ranks_array[0])
    algo2_meanrank = np.mean(ranks_array[1])

    # Return the difference.
    meanrank_diff = algo2_meanrank - algo1_meanrank
    return meanrank_diff

  def _load_metrics_on_permuted_and_diff_rank(self, algo1, algo2,
                                              timeframe_points):
    """Get null distribution on difference in rank, from permuted metric results.

    For across-run metrics (with result_dims 'ATP'), we cannot directly permute
    the metric rankings, because the metrics need to be re-evaluated for each
    permutation of the runs.

    Here we load the metric results that were evaluated on permuted runs
    (permuted between algo1 and algo2). Then, for each permutation, we
    re-compute rankings and differences in mean ranking, in order to obtain a
    null distribution on the difference in mean ranking between algo1 and algo2.

    Args:
      algo1: The first algorithm to be compared.
      algo2: The second algorithm to be compared.
      timeframe_points: List of indices to load, along the "eval points"
        dimension.

    Returns:
      Distribution of differences in mean rank between the two algorithms.
      A 1-D Numpy array with length self.n_random_samples
    """
    assert self.result_dims == 'ATP'  # across-run metrics

    # Load the metrics values
    data_permuted = data_def.DataDefPermuted(
        self.resampled_results_dir,
        algorithms=[algo1, algo2],
        tasks=self.data_def.tasks,
        n_runs_per_experiment=self.data_def.n_runs_per_experiment,
        n_permutation=self.n_random_samples)

    perm_diffs = np.empty(self.n_random_samples)
    for i_perm in range(self.n_random_samples):
      # Load the permuted metric values
      permuted_metric_values = np.empty((2, self.data_def.n_tasks))
      for split in (1, 2):
        for i_task, task in enumerate(self.data_def.tasks):
          all_timepoints = data_permuted.results['%s.seed%d.split%d' %
                                                 (task, i_perm,
                                                  split)][self.metric]
          timeframe_values = np.array(all_timepoints)[timeframe_points]
          permuted_metric_values[split - 1,
                                 i_task] = np.median(timeframe_values)

      # Combine permuted + unpermuted metric values
      other_algos = set(self.data_def.algorithms) - {algo1, algo2}
      unpermuted_metric_values = self.load_metric_results(
          other_algos, timeframe_points)
      all_metric_values = np.concatenate(
          (unpermuted_metric_values, permuted_metric_values), axis=0)

      # Rank the metric values
      metric_ranks_all = self.rank_per_task(all_metric_values)

      # Compute the difference in mean rank, for the algorithms of interest.
      metric_ranks_algo1_algo2 = metric_ranks_all[-2:, :]
      perm_diff = self._algo_meanrank_diff(metric_ranks_algo1_algo2)
      perm_diffs[i_perm] = perm_diff

    return perm_diffs

  def _permute_ranks_and_diff(self, metric_ranks):
    """Get null distribution of differences, by directly permuting ranks.

    Permute ranks across pairs of algorithms, within each task. This allows us
    to obtain a distribution of differences in mean rank

    Args:
       metric_ranks: Array with the rank for each metric result, evaluated
         separately for each task.

    Returns:
      Distribution of differences in mean rank between the two algorithms.
      A 1-D Numpy array with length self.n_random_samples
    """
    # We assume 0th dimension is algorithm and 1st dimension is task.
    assert self.result_dims[:2] == 'AT'
    n_task = metric_ranks.shape[1]

    perm_diffs = np.empty(self.n_random_samples)
    for i_perm in range(self.n_random_samples):

      # Permute across the two algorithms, within each task.
      permuted_ranks = np.zeros(metric_ranks.shape)
      for i_task in range(n_task):
        sample1 = metric_ranks[0, i_task]
        sample2 = metric_ranks[1, i_task]
        all_samples = np.concatenate([sample1, sample2])

        len_sample1 = sample1.shape[0]
        permutation_indices = np.random.permutation(range(len_sample1 * 2))

        permuted_samples = all_samples[permutation_indices]
        permuted_ranks[0, i_task, :] = permuted_samples[:len_sample1]
        permuted_ranks[1, i_task, :] = permuted_samples[len_sample1:]

      # difference between mean ranks, for this permutation
      perm_diff = self._algo_meanrank_diff(permuted_ranks)
      perm_diffs[i_perm] = perm_diff

    return perm_diffs

  @staticmethod
  def _get_pval(null_distribution, observed_value):
    """Compute p-value given an observation and a null distribution.

    I.e. the proportion of the null distribution that is at least as extreme as
    the observed value. Note that this is a two-sided test.

    Args:
      null_distribution: 1-D array, containing the null distribution of the test
        statistic (e.g. on a set of permutations)
      observed_value: float, the observed value of the test statistic

    Returns:
      float, the p-value
    """
    return np.mean(np.abs(null_distribution) >= abs(observed_value))

  def _write_pval_result(self, pval, algo1, algo2, timeframe=None):
    """Write p-value to text file."""
    io_utils.makedirs(self.outfile_dir)
    outfile_path = ('%s/%s_%s_%s' %
                    (self.outfile_dir, self.metric, algo1, algo2))
    if timeframe is not None:
      outfile_path += '_%d' % timeframe

    with open(outfile_path, 'w') as outfile:
      outfile.write('%g' % pval)

    logging.info('P-val result written to: %s', outfile_path)

  def _write_confidence_interval_result(self,
                                        ci_lower,
                                        ci_upper,
                                        algo,
                                        timeframe=None):
    """Write confidence interval to text file."""
    io_utils.makedirs(self.outfile_dir)
    outfile_path = '%s/%s_%s' % (self.outfile_dir, self.metric, algo)
    if timeframe is not None:
      outfile_path += '_%d' % timeframe

    with open(outfile_path, 'w') as outfile:
      outfile.write('%g,%g' % (ci_lower, ci_upper))

    logging.info('Confidence interval written to: %s', outfile_path)
