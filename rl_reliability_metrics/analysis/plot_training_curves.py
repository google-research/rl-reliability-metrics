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

"""Plot training curves."""

import csv
from matplotlib import pyplot as plt
import numpy as np

from rl_reliability_metrics.analysis import io_utils_oss as io_utils
# Internal gfile dependencies

ALGO_COLORS = ('red', 'orange', 'yellow', 'green', 'blue', 'magenta')

# These are maximal performance values reported in the literature.
# go/openai-mujoco-baselines
BASELINES = {
    'ddpg.Ant': 700,
    'ddpg.HalfCheetah': 8500,
    'ddpg.Hopper': 1000,
    'ddpg.Reacher': -5,
    'ddpg.Walker2d': 2300,
    'ppo.HalfCheetah': 1800,
    'ppo.Hopper': 2250,
    'ppo.Reacher': -4,
    'ppo.Swimmer': 105,
    'ppo.Walker2d': 3000,
    'sac.Ant': 6000,
    'sac.HalfCheetah': 15000,
    'sac.Hopper': 3300,
    'sac.Humanoid': 5500,
    'sac.Walker2d': 3700,
    'td3.Ant': 4200,
    'td3.HalfCheetah': 9500,
    'td3.Hopper': 3300,
    'td3.Reacher': -4,
    'td3.Walker2d': 4500,
}


def equalize_axes_for_task(fig_name, xlims_extremes, ylims_extremes,
                           subplot_pos, n_task, n_algo):
  plt.figure(fig_name)
  for subplot_pos in range(subplot_pos - n_algo + 1, subplot_pos + 1):
    plt.subplot(n_task, n_algo, subplot_pos)
    plt.xlim(xlims_extremes)
    plt.ylim(ylims_extremes)


def compute_window_means(curves, window_size):
  """Compute the mean within each window, for each curve.

  Windows are defined as [0, window_size)
                         [window_size, 2*window_size)
                         [2*window_size, 3*window_size)
                         ...

  Args:
    curves: Numpy array [n_curves x 2 x n_timesteps] where curves[:, 0, :]
            are the timesteps and curves[:, 1, :] are the performance values.
    window_size: The size of the window (in timesteps). Set to None if the
      curves are aligned (same timesteps for each curve) and no windowing is
      needed.

  Returns:
    Timesteps (center of each window) [n_windows]
    Window means (mean of each window) [n_curves x n_windows]
  """
  if not window_size:
    # the curves already have aligned timesteps
    timesteps = curves[0, 0, :]  # assume timesteps are same for every curve
    window_means = curves[:, 1, :]
  else:
    max_timestep = np.max(np.hstack(curves[:, 0]))

    window_means = []
    for window_start in range(0, int(max_timestep) + 1, window_size):
      # for each curve, get the mean within the current window
      window_means.append([])
      for curve in curves:
        window_end = window_start + window_size
        window_inds = np.logical_and(curve[0] >= window_start,
                                     curve[0] < window_end)
        window_vals = curve[1][window_inds]
        window_means[-1].append(np.mean(window_vals))

      # get timesteps as center of each window
      timesteps = np.arange(0 + window_size / 2,
                            int(max_timestep + window_size),
                            window_size)  # center of each window
      timesteps = timesteps[:len(window_means)]

    timesteps = np.array(timesteps)
    window_means = np.array(window_means).transpose()

  return timesteps, window_means


def compute_means(window_means):
  """Compute mean across curves.

  Args:
    window_means: Numpy array [n_curves x n_windows]

  Returns:
    Mean across curves. Numpy array [n_windows]
  """
  return np.mean(window_means, axis=0)


def compute_medians(window_means):
  """Compute median across curves.

  Args:
    window_means: Numpy array [n_curves x n_windows]

  Returns:
    Median across curves. Numpy array [n_windows]
  """
  return np.median(window_means, axis=0)


def compute_boot_ci(window_means, n_boot=100):
  """Compute 95% bootstrap confidence intervals across curves.

  Args:
    window_means: Numpy array [n_curves x n_windows]
    n_boot: Number of bootstraps to compute.

  Returns:
    Lower confidence across curves. Numpy array [n_windows]
    Upper confidence across curves. Numpy array [n_windows]
  """
  n_curves = window_means.shape[0]

  boot_vals = []
  for _ in range(n_boot):
    indices = np.random.randint(0, n_curves, n_curves)
    means = compute_means(window_means[indices, :])
    boot_vals.append(means)

  # crop the means sequences to same length
  shortest_len = min([len(means) for means in boot_vals])
  boot_vals = [means[:shortest_len] for means in boot_vals]
  boot_vals = np.array(boot_vals)

  ci_lower = np.percentile(boot_vals, 5, axis=0)
  ci_upper = np.percentile(boot_vals, 95, axis=0)
  return ci_lower, ci_upper


def compute_percentiles(window_means, lower_thresh=5, upper_thresh=95):
  lower = np.percentile(window_means, lower_thresh, axis=0)
  upper = np.percentile(window_means, upper_thresh, axis=0)
  return lower, upper


def plot_baseline(algo, task, color='r'):
  """Plot baseline as a horizontal dashed line."""
  key = '%s.%s' % (algo, task)
  if key in BASELINES:
    baseline = BASELINES[key]
    xlims = plt.xlim()
    plt.hlines(baseline, xlims[0], xlims[1], linestyles='dashed', colors=color)


def update_xylims_extremes(xlims_extremes, ylims_extremes):
  """Update x- and y-lim extremes with current xlim and ylim values."""
  xlims, ylims = plt.xlim(), plt.ylim()
  xlims_extremes = [
      min(xlims[0], xlims_extremes[0]),
      max(xlims[1], xlims_extremes[1])
  ]
  ylims_extremes = [
      min(ylims[0], ylims_extremes[0]),
      max(ylims[1], ylims_extremes[1])
  ]
  return xlims_extremes, ylims_extremes


def fill_plot(x, y, y_lower, y_upper, algo, task, color):
  plt.plot(x, y, color=color)
  plt.fill_between(x, y_lower, y_upper, alpha=0.3, color=color)
  plot_baseline(algo, task, color=color)


def subplots_square(n_subplots):
  """Determine subplot specifications, to achieve square-like plot."""
  n_subplots_x = np.ceil(np.sqrt(n_subplots))
  n_subplots_y = np.ceil(n_subplots / n_subplots_x)
  return n_subplots_x, n_subplots_y


def make_training_curve_plots(algos,
                              tasks,
                              n_runs_per_expt,
                              csv_filepath_template,
                              figure_outdir,
                              window_size=None,
                              subplot_height=5,
                              subplot_width=8):
  """Make four different plots of the training curves.

    (1) Raw training curves, one line per run.
        One subplot per (algo, task) combination.
    (2) Median + 5th/95th percentiles across runs.
        One subplot per (algo, task) combination.
    (3) Median + 5th/95th percentiles across runs.
        One subplot per task (all algos plotted).
    (4) Means + 95% bootstrap CIs across runs.
        One subplot per task (all algos plotted).

  Args:
    algos: List of strings. Which algorithms we are analyzing.
    tasks: List of strings. Which tasks we are analyzing.
    n_runs_per_expt: Number of runs per (algo, task) combination.
    csv_filepath_template: String path to CSV files containing training curves.
      e.g. '/my/path/%s_%s_%d.csv'. Should accept (task, algo, run).
    figure_outdir: Path to directory for saving figures.
    window_size: If the training curves are not aligned (if different curves are
      evaluated at different timepoints), we can specify a window_size that
      aggregates, to allow computing summaries across curves like means,
      medians, percentiles, and confidence intervals.
    subplot_height: Height of each subplot.
    subplot_width: Width of each subplot.
  """
  n_algo = len(algos)
  n_task = len(tasks)

  plt.figure('raw', figsize=(subplot_width * n_algo, subplot_height * n_task))
  plt.figure(
      'medians_percentiles',
      figsize=(subplot_width * n_algo, subplot_height * n_task))
  n_subplots_x, n_subplots_y = subplots_square(n_task)
  plt.figure(
      'medians_percentiles_pertask',
      figsize=(subplot_width * n_subplots_x, subplot_height * n_subplots_y))
  plt.figure(
      'means_CIs_pertask',
      figsize=(subplot_width * n_subplots_x, subplot_height * n_subplots_y))
  fig_names = [
      'raw', 'medians_percentiles', 'medians_percentiles_pertask',
      'means_CIs_pertask'
  ]

  subplot_pos = 0
  # Iterate through each task.
  for i_task, task in enumerate(tasks):
    print('%s...' % task, end='')

    # Initialize x- and y-lims.
    xlims_extremes = [np.inf, -np.inf]
    task_baselines = [
        baseline for key, baseline in BASELINES.items() if task in key
    ]
    if task_baselines:
      ylims_extremes = [np.inf, max(task_baselines)]
    else:
      ylims_extremes = [np.inf, -np.inf]

    # Iterate through each algorithm.
    for i_algo, algo in enumerate(algos):
      subplot_pos += 1
      algo_color = ALGO_COLORS[i_algo]

      plt.figure('raw')
      plt.subplot(n_task, n_algo, subplot_pos)

      # Load and plot the raw curves.
      curves = []
      for run in range(n_runs_per_expt):
        csv_filepath = csv_filepath_template % (task, algo, run)
        with open(csv_filepath, 'r') as csv_file:
          csv_reader = csv.reader(csv_file)
          curve = []
          for _ in range(2):
            curve.append(np.array(csv_reader.next(), dtype=np.float))
        curves.append(curve)
        plt.plot(curve[0], curve[1])
      plot_baseline(algo, task)

      # update the xlim/ylim extremes
      xlims_extremes, ylims_extremes = update_xylims_extremes(
          xlims_extremes, ylims_extremes)

      # Compute summaries
      curves = np.array(curves)
      timesteps, window_means = compute_window_means(curves, window_size)
      means = compute_means(window_means)
      medians = compute_medians(window_means)
      cis = compute_boot_ci(window_means)
      percentiles = compute_percentiles(window_means)

      # plot the medians + percentiles
      plt.figure('medians_percentiles')
      plt.subplot(n_task, n_algo, subplot_pos)
      fill_plot(timesteps, medians, percentiles[0], percentiles[1], algo, task,
                algo_color)

      # Plot the medians + percentiles on a single plot per task.
      plt.figure('medians_percentiles_pertask')
      plt.subplot(n_subplots_y, n_subplots_x, i_task + 1)
      fill_plot(timesteps, medians, percentiles[0], percentiles[1], algo, task,
                algo_color)

      # Plot the mean + CI on a single plot per task.
      plt.figure('means_CIs_pertask')
      plt.subplot(n_subplots_y, n_subplots_x, i_task + 1)
      fill_plot(timesteps, means, cis[0], cis[1], algo, task, algo_color)

      # Figure titles.
      for fig_name in ['raw', 'medians_percentiles']:
        plt.figure(fig_name)
        plt.title('%s - %s' % (algo, task))
    for fig_name in ['medians_percentiles_pertask', 'means_CIs_pertask']:
      plt.figure(fig_name)
      plt.title(task)

    # equalize axes for the task
    for fig_name in ['raw', 'medians_percentiles']:
      equalize_axes_for_task(fig_name, xlims_extremes, ylims_extremes,
                             subplot_pos, n_task, n_algo)

  # Add legends
  for fig_name in ['medians_percentiles_pertask', 'means_CIs_pertask']:
    plt.figure(fig_name)
    plt.legend(algos)

  # Save the figures.
  io_utils.makedirs(figure_outdir)
  for fig_name in fig_names:
    plt.figure(fig_name)
    plt.tight_layout()
    output_path = '%s/%s.png' % (figure_outdir, fig_name)
    with open(output_path, 'wb') as outfile:
      plt.savefig(outfile, dpi=100)
