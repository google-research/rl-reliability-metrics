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

"""Class for making plots of robustness metric results and statistics."""

import datetime
import math
import os

from absl import logging
from matplotlib import pyplot as plt
import numpy as np
from rl_reliability_metrics.analysis import io_utils_oss as io_utils
from rl_reliability_metrics.analysis import plot_utils
from rl_reliability_metrics.analysis import stats
from rl_reliability_metrics.analysis import stats_utils

# Internal gfile dependencies

HATCH_PATTERNS = ('-', '/', '.', 'O', '+', 'o', 'x', '*', '\\')
ALGO_COLORS = ('r', 'y', 'g', 'b', 'm')
MARKERS = ('o', 's', 'v', '^', '<', '>')
TIMEFRAME_NAMES = ['Beginning', 'Middle', 'End']
UP_ARROW = r' $\uparrow$'
DOWN_ARROW = r' $\downarrow$'


class Plotter(object):
  """Class for making plots of metric results and statistics."""

  def __init__(self,
               data,
               pvals_dir,
               confidence_intervals_dir,
               n_timeframes,
               algorithms=None,
               out_dir=None,
               pthresh=0.01,
               multiple_comparisons_method='benjamini-yekutieli',
               subplot_axis_labels=True,
               make_legend=False):
    """Initialize Plotter object.

    Args:
      data: DataDef object containing all the metric results.
      pvals_dir: Path to directory containing p-values for comparisons between
        pairs of algorithms.
      confidence_intervals_dir: Path to directory containing bootstrap
        confidence intervals.
      n_timeframes: Total number of timeframes we are dividing each run into.
      algorithms: If specified, these algorithms will be plotted, in this order.
        If None, we plot all algorithms available in the data (order not
        guaranteed).
      out_dir: Path to directory where we save the plot images. If None, we
        simply display the images without saving.
      pthresh: p-value threshold for significance.
      multiple_comparisons_method: String indicating method to use for multiple
        comparisons correction. See self._do_multiple_comparisons_correction for
        options.
      subplot_axis_labels: Whether to add x- and y-axis labels for each subplot.
      make_legend: Whether to make a legend.
    """
    self.data_def = data
    self.pvals_dir = pvals_dir
    self.confidence_intervals_dir = confidence_intervals_dir
    self.n_timeframes = n_timeframes
    self.out_dir = out_dir
    self.pthresh = pthresh
    self.multiple_comparisons_method = multiple_comparisons_method
    self.subplot_axis_labels = subplot_axis_labels
    self.make_legend = make_legend

    # Parse information from data_def
    self.dataset = self.data_def.dataset
    self.algorithms = algorithms if algorithms else self.data_def.algorithms
    self.n_algo = len(self.algorithms)
    self.n_task = len(self.data_def.tasks)

    # Bonferroni-corrected p-value threshold
    self.pthresh_corrected = stats_utils.multiple_comparisons_correction(
        self.n_algo, self.pthresh, self.multiple_comparisons_method)

  def make_plots(self, metric):
    """Make all plots for a given metric.

    Args:
      metric: String name of the metric.
    """
    plot_utils.paper_figure_configs()

    # Create a metric-specific StatsRunner object
    stats_runner = stats.StatsRunner(self.data_def, metric, self.n_timeframes)

    result_dims = stats_runner.result_dims
    if result_dims == 'ATRP':
      # Within-runs metric with eval points.
      self._make_plots_with_eval_points(metric, stats_runner)
    elif result_dims == 'ATR':
      # Within-runs metrics without eval points (one value per run).
      self._make_plots_no_eval_points(metric, stats_runner)
    elif result_dims == 'ATP':
      # Across-runs metric with eval points
      self._make_plots_with_eval_points(metric, stats_runner)
    else:
      raise ValueError('plotting not implemented for result_dims: %s' %
                       result_dims)

  def _save_fig(self, metric, plot_name):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filepath = os.path.join(self.out_dir,
                            '%s__%s__%s.png' % (metric, plot_name, timestamp))
    io_utils.makedirs(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
      plt.savefig(f)
    logging.info('Plot output to: %s', filepath)

  def _make_plots_with_eval_points(self, metric, stats_runner):
    """Make plots for a metric evaluated at multiple evaluation points per run.

    e.g. 'ATP' or 'ATRP' metrics.

    Plot 1: raw metric values per task.
    * One subplot per task.
    * Each subplot contains a plot showing the metric values across evaluation
      points. For ATRP metrics, we show the median metric values and fill plots
      indicating the IQR at each evaluation point.

    Plot 2: Mean rankings across tasks.
    * One subplot per timeframe.
    * One bar plot showing the mean ranking for each algorithm, and horizontal
      line segments indicating which pairs of algorithms are statistically
      different.

    Args:
      metric: String specifying the metric.
      stats_runner: StatsRunner object
    """
    # Set up figure for per-task raw values.
    subplot_ncol_1 = 4
    n_subplots_1 = self.n_task + 1 if self.make_legend else self.n_task
    subplot_nrow_1 = math.ceil(n_subplots_1 / subplot_ncol_1)
    fig1 = plt.figure(figsize=(4 * subplot_ncol_1, 4 * subplot_nrow_1))

    # Set up figure for mean rankings.
    subplot_ncol_2 = self.n_timeframes
    if self.make_legend:
      subplot_ncol_2 += 1
    subplot_nrow_2 = 1
    fig2 = plt.figure(figsize=(4 * subplot_ncol_2, 4 * subplot_nrow_2))

    ##=== Plot 1: Raw metric values per task ===##

    plt.figure(fig1.number)
    eval_point_idxs = stats_runner.get_timeframe_points(None)
    eval_point_values = self.data_def.metric_params[metric]['eval_points']
    metric_results = stats_runner.load_metric_results(
        self.algorithms, eval_point_idxs, collapse_on_timepoints=False)
    result_dims = stats_runner.result_dims

    for i_task in range(self.n_task):
      plt.subplot(subplot_nrow_1, subplot_ncol_1, i_task + 1)
      task_results = np.squeeze(metric_results[:, i_task])
      if len(eval_point_idxs) == 1:
        task_results = np.expand_dims(task_results, -1)

      if result_dims == 'ATP':
        # For across-run metrics, we plot a single curve.
        for i_algo in range(self.n_algo):
          plt.plot(eval_point_values, task_results[i_algo, :],
                   marker=MARKERS[i_algo])
        if self.subplot_axis_labels:
          plt.xlabel('evaluation points', fontsize=16)
          plt.ylabel('metric values', fontsize=16)

      elif result_dims == 'ATRP':
        # For per-run metrics, we plot the median and IQR across curves.
        for i_algo in range(self.n_algo):
          algo_color = ALGO_COLORS[i_algo]
          task_algo_results = task_results[i_algo]  # n_runs x n_eval_points
          result_medians = np.median(task_algo_results, axis=0)
          result_quartile1 = np.percentile(task_algo_results, q=25, axis=0)
          result_quartile3 = np.percentile(task_algo_results, q=75, axis=0)
          plt.plot(eval_point_values, result_medians, algo_color,
                   marker=MARKERS[i_algo])
          plt.fill_between(
              eval_point_values,
              result_quartile1,
              result_quartile3,
              alpha=0.3,
              color=algo_color)
        if self.subplot_axis_labels:
          plt.xlabel('evaluation points', fontsize=16)
          plt.ylabel('metric values', fontsize=16)

      else:
        raise ValueError('result_dims must be ATP or ATRP, not %s' %
                         result_dims)

      plot_utils.simple_axis(plt.gca())
      plt.title(self.data_def.tasks[i_task])

    # plot the legend
    if self.make_legend:
      plt.subplot(subplot_nrow_1, subplot_ncol_1, n_subplots_1)
      self._lineplot_legend()

    ##=== Plot 2: Mean rankings (mean across tasks) ===##

    for timeframe in range(self.n_timeframes):
      # Load data for plotting.
      timeframe_points = stats_runner.get_timeframe_points(timeframe)
      pvals = self._load_pvals(metric, timeframe)
      confidence_intervals = self._load_confidence_intervals(
          metric, stats_runner, timeframe)

      plt.figure(fig2.number)
      metric_results = stats_runner.load_metric_results(
          self.algorithms, timeframe_points, collapse_on_timepoints=True)
      plt.subplot(subplot_nrow_2, subplot_ncol_2, timeframe + 1)
      self._plot_bars_and_significant_differences(metric_results, pvals,
                                                  confidence_intervals,
                                                  stats_runner)
      plt.title(TIMEFRAME_NAMES[timeframe], fontsize=14)

    # plot the legend
    if self.make_legend:
      plt.subplot(subplot_nrow_2, subplot_ncol_2, subplot_ncol_2)
      self._barplot_legend()

    ##=== Wrap up the figures ===##

    for fig, plot_name in [(fig1, 'per-task_raw'), (fig2, 'mean_rankings')]:
      if plot_name == 'per-task_raw':
        suptitle_suffix = (
            UP_ARROW if stats_runner.bigger_is_better else DOWN_ARROW)
      else:
        suptitle_suffix = ''
      plt.figure(fig.number, plot_name)
      self._wrap_up_figure(metric, plot_name, suptitle_suffix)

  def _make_plots_no_eval_points(self, metric, stats_runner):
    """Make plots for a metric without evaluation points (one value per run).

    e.g. 'ATR' metrics.

    Plot 1: Raw metric values per task.
    * One subplot per task.
    * Each subplot contains a box-and-whisker plot showing the median metric
     values for each algorithm, a box indicating 1st and 3rd quartiles, and
     whiskers indicating the minimum and maximum values (excluding outliers,
     defined as being outside 1.5x the inter-quartile range from the 1st and 3rd
     quartiles).

    Plot 2: Mean rankings across tasks.
    * One bar plot showing the mean ranking for each algorithm, and horizontal
      line segments indicating which pairs of algorithms are statistically
      different.

    Args:
      metric: String specifying the metric.
      stats_runner: StatsRunner object
    """

    # Load data for plotting.
    metric_results = stats_runner.load_metric_results(
        self.algorithms, timeframe_points=None)
    pvals = self._load_pvals(metric)
    confidence_intervals = self._load_confidence_intervals(metric, stats_runner)

    ##=== Plot 1: Raw metric values per task ===##

    # Set up figure.
    subplot_ncol = 4
    n_subplot = self.n_task
    if self.make_legend:
      n_subplot += 1
    subplot_nrow = math.ceil(n_subplot / subplot_ncol)
    plt.figure(figsize=(4 * subplot_ncol, 4 * subplot_nrow))

    # Plot the raw metric values as box-and-whisker plots.
    for i_task in range(self.n_task):
      plt.subplot(subplot_nrow, subplot_ncol, i_task + 1)
      task_results = np.squeeze(metric_results[:, i_task, :])
      boxplot = plt.boxplot(task_results.T, patch_artist=True)
      for part in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(boxplot[part], color='k')
      for i_patch, patch in enumerate(boxplot['boxes']):
        patch.set(facecolor=ALGO_COLORS[i_patch])
      plt.title(self.data_def.tasks[i_task], fontsize=16)
      self._configure_axes('Raw metric values')
      self._extend_ylims_past_zero(task_results)
      plot_utils.simple_axis(plt.gca())

    if self.make_legend:
      plt.subplot(subplot_nrow, subplot_ncol, n_subplot)
      self._barplot_legend()

    # Wrap up the figure.
    suptitle_suffix = (
        UP_ARROW if stats_runner.bigger_is_better else DOWN_ARROW)
    self._wrap_up_figure(
        metric, plot_name='per-task_raw', suptitle_suffix=suptitle_suffix)

    ##=== Plot 2: Mean rankings (mean across tasks) ===##

    # Set up figure.
    subplot_ncol = 2 if self.make_legend else 1
    subplot_nrow = 1
    plt.figure(figsize=(4 * subplot_ncol, 4 * subplot_nrow))

    # Plot mean rankings and show statistical differences
    plt.subplot(subplot_nrow, subplot_ncol, 1)
    self._plot_bars_and_significant_differences(metric_results, pvals,
                                                confidence_intervals,
                                                stats_runner)
    plot_utils.simple_axis(plt.gca())

    # plot the legend
    if self.make_legend:
      plt.subplot(subplot_nrow, subplot_ncol, subplot_ncol)
      self._barplot_legend()

    # Wrap up the figure.
    self._wrap_up_figure(metric, plot_name='mean_rankings')

  def _wrap_up_figure(self, metric, plot_name, suptitle_suffix=''):
    """Add suptitle, set tight layout, and save the figure."""
    plt.suptitle(
        plot_utils.METRICS_DISPLAY_NAMES[metric] + suptitle_suffix, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if self.out_dir:
      self._save_fig(metric, plot_name)

  def _load_pvals(self, metric, timeframe=None):
    """Load previously computed p-values.

    Args:
      metric: Which metric we are plotting.
      timeframe: Which timeframe we are plotting. Set None if irrelevant (for
        metrics that are not evaluated at specific eval points).

    Returns:
      Dictionary of p-values, with entries {'algo1.algo2': pval}
    """
    pvals = {}
    for algo1 in self.algorithms:
      for algo2 in self.algorithms:
        # Get path to p-value
        pvals_filepath = ('%s/%s_%s_%s' %
                          (self.pvals_dir, metric, algo1, algo2))
        if timeframe is not None:
          pvals_filepath += '_%d' % timeframe

        # Load the p-value
        with open(pvals_filepath, 'r') as f:
          pval = float(f.readline())
        pvals['%s.%s' % (algo1, algo2)] = pval
    logging.info('P-values loaded:')
    logging.info(pvals)

    return pvals

  def _load_confidence_intervals(self, metric, stats_runner, timeframe=None):
    """Load previously computed confidence intervals.

    Args:
      metric: Which metric we are plotting.
      stats_runner: StatsRunner object
      timeframe: Which timeframe we are plotting. Set None if irrelevant (for
        metrics that are not evaluated at specific eval points).

    Returns:
      Dictionary of confidence intervals, with entries
      {'algo': [ci_lower, ci_upper]}
    """
    cis = {}
    for algo in self.algorithms:
      # Get path to confidence intervals
      ci_filepath = '%s/%s_%s' % (self.confidence_intervals_dir, metric, algo)
      if timeframe is not None:
        ci_filepath += '_%d' % timeframe

      # Load the p-value
      with open(ci_filepath, 'r') as f:
        line = f.readline()
      ci = list(map(float, line.split(',')))

      # Normalize to range (1, n_metrics)
      if 'R' in stats_runner.result_dims:
        ci[0] /= self.data_def.n_runs_per_experiment
        ci[1] /= self.data_def.n_runs_per_experiment

      cis[algo] = ci
    logging.info('Confidence intervals loaded:')
    logging.info(cis)

    return cis

  def _plot_bars_and_significant_differences(self, metric_results, pvals,
                                             confidence_intervals,
                                             stats_runner):
    """For a single timeframe, plot mean rank and show significant differences.

    Args:
      metric_results: Numpy array with metric values. First two dimensions
        should be (n_algorithm, n_task)
      pvals: p-values on comparison between each pair of algorithms. A dict with
        entries {'algo1.algo2': pvalue}.
      confidence_intervals: Confidence intervals on mean rank for each
        algorithm. A dict with entries {'algo': [ci_lower, ci_upper]}.
      stats_runner: StatsRunner object
    """
    ymax = 1.32 * (len(self.algorithms))
    y_pval_lines = 0.83

    # First get the rankings across all algos
    metric_ranks = stats_runner.rank_per_task(metric_results)

    # Get mean ranks over tasks, for each algo
    # (collapse across all other dimensions)
    extra_dims = range(1, len(metric_ranks.shape))
    mean_ranks = np.mean(metric_ranks, tuple(extra_dims))

    # Normalize the ranks to range (1, n_algorithms)
    if 'R' in stats_runner.result_dims:
      mean_ranks /= self.data_def.n_runs_per_experiment

    # Plot the mean rankings and error bars for each algo
    for i_algo, algo in enumerate(self.algorithms):
      plot_utils.flipped_errorbar(
          x=i_algo,
          y=mean_ranks[i_algo],
          yerr=confidence_intervals[algo],
          ymax=self.n_algo,
          bar_color=ALGO_COLORS[i_algo],
          hatch_pattern=HATCH_PATTERNS[i_algo],
          x_offset=0.6,
      )

    # Rank order the p-values.
    if self.multiple_comparisons_method != 'bonferroni':
      # Get subset of the p-values: we don't need the reverse comparisons, and
      # we don't need the self comparisons.
      pvals_subset = {}
      for i_algo, algo1 in enumerate(self.algorithms):
        for j_algo in range(i_algo + 1, self.n_algo):
          algo2 = self.algorithms[j_algo]
          algo_str = '%s.%s' % (algo1, algo2)
          pvals_subset[algo_str] = pvals[algo_str]

      sorted_keys = sorted(pvals_subset, key=pvals_subset.get)
      pval_ranks = {key: rank for rank, key in enumerate(sorted_keys)}

    # Plot black bars indicating significant differences.
    n_lines_plotted = 0
    for i_algo, algo1 in enumerate(self.algorithms):
      for j_algo in range(i_algo + 1, self.n_algo):
        algo2 = self.algorithms[j_algo]
        algo_pair_str = '%s.%s' % (algo1, algo2)

        if self.multiple_comparisons_method != 'bonferroni':
          pval_rank = pval_ranks[algo_pair_str]
          pthresh_corrected = self.pthresh_corrected[pval_rank]
        else:
          pthresh_corrected = self.pthresh_corrected

        if pvals[algo_pair_str] < pthresh_corrected:
          x = [i_algo + 1, j_algo + 1]
          y = [(y_pval_lines + n_lines_plotted * 0.03) * ymax] * 2
          plt.plot(x, y, color='k')
          n_lines_plotted += 1

    self._configure_axes('normalized mean rank', range(1, self.n_algo + 1),
                         range(self.n_algo, 0, -1))

  def _configure_axes(self, y_label, y_ticks=None, y_tick_labels=None):
    """Configure axis limits and labels."""
    algo_abbreviations = [
        plot_utils.ALGO_ABBREVIATIONS[algo] for algo in self.algorithms
    ]
    plt.xticks(range(1, self.n_algo + 1), algo_abbreviations)
    plt.xlim(0, len(self.algorithms) + 1)
    if y_ticks:
      plt.yticks(y_ticks)
    if y_tick_labels:
      plt.gca().set_yticklabels(y_tick_labels)
    if self.subplot_axis_labels:
      plt.xlabel('algorithm', fontsize=16)
      plt.ylabel(y_label, fontsize=16)
    plt.tick_params(top='off')

  @staticmethod
  def _extend_ylims_past_zero(data, tolerance=0.01, extension=0.1):
    """Extend y-axis to ensure that zero-values in the data are visible.

    Args:
      data: Data being plotted.
      tolerance: Determines what values are considered too close to zero.
      extension: Determines how far to extend the y-axis.
    """
    ylims_orig = plt.gca().get_ylim()
    abs_min = np.abs(np.min(data))
    abs_max = np.abs(np.max(data))

    # Extend below zero.
    if abs_min < tolerance * abs_max:
      ylim_lower = -ylims_orig[1] * extension
      plt.ylim([ylim_lower, ylims_orig[1]])

    # Extend above zero.
    elif abs_max < tolerance * abs_min:
      ylim_upper = -ylims_orig[0] * extension
      plt.ylim([ylims_orig[0], ylim_upper])

  def _barplot_legend(self):
    """Plot a legend showing the color/texture for each algorithm."""
    for ibox in range(self.n_algo):
      box_y = self.n_algo - ibox
      plt.scatter(
          0,
          box_y,
          s=300,
          marker='s',
          facecolor=ALGO_COLORS[ibox],
          edgecolor='k',
          hatch=HATCH_PATTERNS[ibox],
          label=HATCH_PATTERNS[ibox])
      plt.text(0.008, box_y - 0.15, self.algorithms[ibox], fontsize=14)
      plt.xlim(-0.01, 0.05)

    plot_utils.no_axis(plt.gca())

  def _lineplot_legend(self):
    """Plot a legend showing the color/marker for each algorithm."""
    for i_algo in range(self.n_algo):
      y = self.n_algo - i_algo
      color = ALGO_COLORS[i_algo]
      plt.plot([0, 2], [y, y], color=color)
      plt.plot(1, y, marker=MARKERS[i_algo], color=color)
      plt.text(2.5, y - 0.002, self.algorithms[i_algo], fontsize=14)
    ax = plt.gca()
    plot_utils.no_axis(ax)
    ax.set_axis_bgcolor('white')
    plt.xlim([0, 10])
    plt.ylim([0, self.n_algo + 1])

