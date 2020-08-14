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

"""Utilities for computing metrics."""

import copy
import functools

from absl import logging

import numpy as np
import scipy.signal


def assert_non_empty(curves):
  """Raises ValueError if curves is an empty list."""
  if not curves:
    raise ValueError(
        'Unsupported value of curves: %s. Must be a non-empty list.' % curves)


def get_all_valid_eval_points(curves, window_size=1):
  """Selects timepoints common to all curves and which have valid windows."""
  timepoints_all_curves = [curve[0] for curve in curves]
  common_timepoints = functools.reduce(np.intersect1d, timepoints_all_curves)
  min_eval_point = np.min(common_timepoints) + window_size - 1
  eval_points = common_timepoints[common_timepoints >= min_eval_point]
  return eval_points


def median_absolute_deviations(values, axis=None):
  """Computes median absolute deviation (MAD).

  Args:
    values: 1-D or 2-D Numpy array
    axis: Axis along which to compute MAD. If None, compute MAD on flattened
      version of the array.

  Returns:
    Median absolute deviation.
  """
  if axis not in (None, 0, 1):
    raise ValueError('axis must be in (None, 0, 1).')
  if len(values.shape) > 2:
    raise ValueError('values must be a 1-D or 2-D Numpy array.')

  medians = np.median(values, axis)
  if axis is None:
    absolute_deviations = np.abs(values - medians)
  elif axis == 0:
    nrow = values.shape[0]
    absolute_deviations = np.abs(values - np.tile(medians, (nrow, 1)))
  elif axis == 1:
    ncol = values.shape[1]
    absolute_deviations = np.abs(values - np.tile(medians, (1, ncol)))
  return np.median(absolute_deviations, axis)


def subtract_baseline(curves, baseline):
  """Subtracts a baseline value from a set of curves.

  Args:
    curves: A list of learning curves, each a 2D numpy array where curve[0, :]
      is the timepoint variable and curve[1, :] is the dependent variable.
    baseline: Float value to subtract from the dependent variable on each curve.

  Returns:
    A list of learning curves, with the same format as the original curves.
  """

  return apply_fn_to_curves(curves, lambda curve: curve - baseline)


def divide_by_baseline(curves, baselines):
  """Divides each curve in a set of curves by a baseline value.

  Args:
    curves: A list of learning curves, each a 2D numpy array where curve[0, :]
      is the timepoint variable and curve[1, :] is the dependent variable.
    baselines: Float value to be divided from the dependent variable on each
      curve. Or a list of float values, with the same length as curves.

  Returns:
    A list of learning curves, with the same format as the original curves.
  """

  if isinstance(baselines, list) or isinstance(baselines, tuple) or isinstance(
      baselines, np.ndarray):
    curves_final = []
    for curve, baseline in zip(curves, baselines):
      curves_final.extend(apply_fn_to_curves([curve], lambda c: c / baseline))  # pylint: disable=cell-var-from-loop
  else:
    curves_final = apply_fn_to_curves(curves, lambda c: c / baselines)
  return curves_final


def band_normalization(curves, low, high):
  """Normalizes a set of curves using upper and lower baselines.

  normalized_y  = (y - low) / (high - low)

  Args:
    curves: A list of learning curves, each a 2D numpy array where curve[0, :]
      is the timepoint variable and curve[1, :] is the dependent variable.
    low: Float value representing the lower baseline.
    high: Float value representing the upper baseline.

  Returns:
    A list of normalized curves, with the same format as the original curves.
  """
  if high == low:
    logging.warning(
        'high == low (%0.3g). This will lead '
        'to dividing by zero.', low)

  def normalize_curve(curve):
    return (curve - low) / (high - low)

  return apply_fn_to_curves(curves, normalize_curve)


def mean_normalization(curves):
  """Normalizes a set of curves by the absolute value of the mean.

   For each curve, we normalize the dependent variables using
   curve / abs(curve_mean).

  Args:
    curves: A list of learning curves, each a 2D numpy array where curve[0, :]
      is the timepoint variable and curve[1, :] is the dependent variable.

  Returns:
    A list of learning curves, with the same format as the original curves.
  """
  curve_abs_means = np.array([np.abs(np.mean(curve[1, :])) for curve in curves])
  curves_new = [
      np.array([curve[0, :], curve[1, :] / curve_abs_mean])
      for curve, curve_abs_mean in zip(curves, curve_abs_means)
  ]
  return curves_new


def median_rollout_performance(rollout_sets):
  """Compute the median rollout performance for each rollout set.

  Args:
    rollout_sets: A list of rollout sets, with length n_rollout_sets.
      Each element of the list corresponds to the performance values of one
      set of rollouts that we will measure the median across (e.g. for a
      single model checkpoint). It is a 2D numpy array where rollouts[0, :] is
      just an index variable (e.g. range(0, n_rollouts)) and rollouts[1, :]
      are the performances per rollout.

  Returns:
     Median performance across rollouts, computed for each rollout set.
     (1-D Numpy array with length = n_rollout_sets)
  """
  median_performances = []
  for rollout_set in rollout_sets:
    median = np.median(rollout_set[1, :])
    median_performances.append(median)
  return median_performances


def differences(curves):
  """Computes normalized 1st-order differences (e.g. for detrending).

  To handle curves with uneven spacing between timepoints, and to ensure
  invariance to evaluation frequency, we divide the differences in the dependent
  variable by the differences in the timepoint variable: delta_y / delta_t.

  Args:
    curves: A list of learning curves, each a 2D numpy array where curve[0, :]
      is the timepoint variable and curve[1, :] is the dependent variable.

  Returns:
    A list of diff_curves, each a 2D numpy array with length 1 shorter than the
    original curves. diff_curve[0, :] are the timepoints (with the first
    timepoint removed relative to the original curves), and diff_curve[1, :]
    are the differences.
  """
  diff_curves = []
  for curve in curves:
    new_timepoints = curve[0, 1:]  # shorten by 1
    x_diffs = np.diff(curve[0, :])
    y_diffs = np.diff(curve[1, :])
    diffs = np.true_divide(y_diffs, x_diffs)
    diff_curves.append(np.array([new_timepoints, diffs]))
  return diff_curves


def compute_drawdown(sequence):
  """Computes the drawdown for a sequence of numbers.

    The drawdown at time T is the decline from the highest peak occurring at or
    before time T. https://en.wikipedia.org/wiki/Drawdown_(economics).

    The drawdown is always non-negative. A larger (more positive) drawdown
    indicates a larger drop.

  Args:
    sequence: A numpy array.

  Returns:
    A numpy array of same length as the original sequence, containing the
      drawdown at each timestep.
  """
  peak_so_far = np.maximum.accumulate(sequence)
  return peak_so_far - sequence


def compute_cvar(array, tail, alpha):
  if tail == 'lower':
    value_at_risk = scipy.stats.scoreatpercentile(array, 100 * alpha)
    cvar = array[array <= value_at_risk].mean()
  elif tail == 'upper':
    value_at_risk = scipy.stats.scoreatpercentile(array, 100 * (1 - alpha))
    cvar = array[array >= value_at_risk].mean()
  return cvar


def across_runs_preprocess(curves, eval_points, window_size, lowpass_thresh):
  """Perform preprocessing steps before evaluating across-run metrics.

  1. Determine all valid eval points, if not given.
  2. Low-pass filter, if specified.
  3. If window_size is specified, define windows centered on each eval point,
     and take the curve values at the point closest to the center of each
     window (and within the window). Else just take the curve values at the
     eval points.
  4. Get the curve values at the eval points.

  Args:
    curves: A list of learning curves, each a 2D numpy array where curve[0, :]
      is the timepoint variable and curve[1, :] is the dependent variable.
    eval_points: We return the values of the curves at these timepoints (or the
      timepoints closest to these points, if window_size is defined). If
      eval_points = None, we first find all valid eval points.
    window_size: If not None, we evaluate our curves at the points closest to
      the eval_points (but still within a window centered on the eval_points).
    lowpass_thresh: If not None, we perform low-pass filtering using this
      threshold.

  Returns:
    Numpy array containing preprocessed curves, dependent variables only.
    [n_curves x n_eval_points]
  """
  # Determine eval points.
  eval_points_new = copy.deepcopy(eval_points)
  if eval_points is None:
    eval_points_new = get_all_valid_eval_points(curves, window_size=1)

  # Low-pass filter.
  if not lowpass_thresh:
    curves_filtered = curves
  else:
    curves_filtered = lowpass_filter(curves, lowpass_thresh)

  if window_size:
    # Get curve values at the timepoints closest to each eval point (but within
    # a window centered on the eval point).
    eval_point_vals = get_nearest_within_window(curves_filtered,
                                                eval_points_new, window_size)
  else:
    # Get curve values at the eval points.
    eval_point_vals = apply_window_fn(curves_filtered, eval_points_new,
                                      lambda x: x, 1)

  return eval_point_vals


def lowpass_filter(curves, lowpass_thresh):
  filt_b, filt_a = scipy.signal.butter(8, lowpass_thresh)

  def butter_filter_fn(curve):
    padlen = min(len(curve) - 1, 3 * max(len(filt_a), len(filt_b)))
    return scipy.signal.filtfilt(filt_b, filt_a, curve, padlen=padlen)

  curves_filtered = apply_fn_to_curves(curves, butter_filter_fn)
  return curves_filtered


def find_nearest(array, targets):
  """Find the value in the array that is closest to each target value.

  If two values in the array are the same distance from a target, we return the
  first one.

  Args:
    array: 1-D numpy array
    targets: 1-D numpy array of target values

  Returns:
    * The values in the array that are closest to each target value
    * The distance from the target to each of the returned values
    Both arrays have length len(targets).
  """
  diffs = np.abs([array - target for target in targets])
  idx = diffs.argmin(axis=1)
  nearest_values = np.array([array[i] for i in idx])
  distances = np.abs(nearest_values - targets)
  return nearest_values, distances


def get_nearest_within_window(curves, eval_points, window_size):
  """Get the values of the curves at the timepoints closest to the eval points.

  And within a window of window_size centered on the eval points.

  Args:
    curves: list of curves (2-D arrays where curve[0] is the timepoints and
      curve[1] are the curve values at those timepoints)
    eval_points: 1-D numpy array
    window_size: int

  Returns:
    The curve values closest to the eval points. array [n_curve x n_eval_points]

  Raises:
    ValueError if there exist curves and eval points for which there are no
    points within a window centered on the eval point.
  """
  eval_point_vals = []
  for icurve, curve in enumerate(curves):

    # Find the closest timepoints
    timepoints = curve[0, :]
    eval_points_windowed, distances = find_nearest(timepoints, eval_points)

    # Check that they are all within the windows
    half_window = np.float(window_size) / 2
    if np.any(distances >= half_window):
      offending_eval_points = eval_points[distances >= half_window]
      raise ValueError('For curve %d, there is no timepoint within window '
                       'of size %d around eval point %d' %
                       (icurve, window_size, offending_eval_points))

    # Get the curve values at those timepoints
    curve_vals_at_timepoints = apply_window_fn([curve], eval_points_windowed,
                                               lambda x: x)[0, :]

    eval_point_vals.append(curve_vals_at_timepoints)

  eval_point_vals = np.array(eval_point_vals)
  return eval_point_vals


def curve_range(curves):
  """Compute the range of each curve.

  This is defined as the 95th percentile - the start value.

  Args:
    curves: A list of learning curves, each a 2D numpy array where curve[0, :]
      is the timepoint variable and curve[1, :] is the dependent variable.

  Returns:
    A Numpy array of length len(curves), containing the range of each curve as
    defined above.
  """
  ranges = np.empty(len(curves))
  for icurve, curve in enumerate(curves):
    dependent_variable = curve[1, :]
    curve_95th = np.percentile(dependent_variable, 95)
    curve_start = dependent_variable[0]
    ranges[icurve] = curve_95th - curve_start
  return ranges


def apply_fn_to_curves(curves, dependent_var_fn, independent_var_trim=None):
  """Apply function to the dependent variable of each curve in a list of curves.

  Optionally, trim the independent variable, e.g. to match the length of the
  dependent variable.

  Args:
    curves: A list of learning curves, each a 2D numpy array where curve[0, :]
      is the timepoint variable and curve[1, :] is the dependent variable.
    dependent_var_fn: A function that can be applied to a 1D numpy array, and
      which will be applied to the dependent variable of each curve.
    independent_var_trim: Specify handling of the independent variable. - Set
      'drop' to entirely drop the independent variable (returning only a
      1D-array for each curve, containing the processed dependent variable.) -
      Set to a list of integers to indicate a specific set of indices to trim
      from the independent variable of each curve. E.g. set [0] to trim the
      first element, and set [-1, -2] to trim the last 2 element. - Set None for
      no trimming.

  Returns:
    A list of processed curves, each curve a numpy array.

    If 'independent_var_trim' is not 'drop', then each curve is a 2D numpy
    array. For each curve, curve[0, :] are the timepoints (potentially with some
    elements trimmed compared to the original curves), and curve[1, :] is the
    result of the specified function applied to the dependent variable of the
    original curves.

    If 'independent_var_trim' is 'drop', then each curve is a 1D numpy array,
    the result of the specified function applied to the dependent variable of
    the original curves.
  """
  curves_new = []
  for curve in curves:
    if independent_var_trim == 'drop':
      curves_new.append(dependent_var_fn(curve[1, :]))
    else:
      curve_indices = range(curve.shape[1])
      if independent_var_trim is not None:
        curve_indices = np.delete(curve_indices, independent_var_trim)
      curves_new.append(
          np.array([curve[0, curve_indices],
                    dependent_var_fn(curve[1, :])]))
  return curves_new


def apply_window_fn(curves, eval_points, window_fn, window_size=1):
  """Checks for invalid curves and applies a window_fn to the curves.

  Applies the window_fn using window_size at eval_points to the curves.

  Args:
    curves: A list of learning curves, each a 2D numpy array where curve[0, :]
      is the timepoint variable and curve[1, :] is the dependent variable.
    eval_points: The values of the curves will be returned at these timepoints.
    window_fn: What function to apply to each window.
    window_size: The function takes a window of length window_size at each of
      the eval_points, i.e. [timepoint - window_size + 1, timepoint], and
      applies the window_fn in this window.

  Returns:
    A 2-D Numpy array of curves, only at the eval_points.
    [ncurves x n_eval_points]

  Raises:
    ValueError: if the input `curves` is empty, if the windows are out of
    bounds on any of the curves, or if any of the windows are empty.
  """
  assert_non_empty(curves)

  # Apply the window_fn at the eval_points.
  curve_points = np.empty((len(curves), len(eval_points)))
  for c, curve in enumerate(curves):
    timepoint_vals = curve[0, :]
    dependent_vals = curve[1, :]

    for e, eval_point in enumerate(eval_points):
      # Get window timepoints.
      window_timepoints = timepoint_vals[np.logical_and(
          timepoint_vals >= eval_point - window_size + 1,
          timepoint_vals <= eval_point)]

      # Check if window is empty.
      if len(window_timepoints) == 0:  # pylint: disable=g-explicit-length-test
        raise ValueError(
            'No timepoints exist in window of size %d at eval_point %s for '
            'curve %d of %d: %s' %
            (window_size, eval_point, c, len(curves), curve))

      # Apply window_fn to the window.
      window_indices = [
          ind for ind, timepoint in enumerate(timepoint_vals)
          if timepoint in window_timepoints
      ]
      curve_points[c, e] = window_fn(dependent_vals[window_indices])

  return curve_points
