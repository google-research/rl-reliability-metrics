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

"""Online metrics for evaluating robustness of an RL algorithm.

Given a learning curve or set of learning curves, these metrics provide
measures of the robustness of the RL algorithm.
"""

import abc
import copy
import functools
import gin
import numpy as np
from rl_reliability_metrics.metrics import metric_utils as utils
from rl_reliability_metrics.metrics import metrics_base
import scipy.signal
import scipy.stats
import six


@six.add_metaclass(abc.ABCMeta)
class _OnlineMetric(metrics_base.Metric):
  """Base class for online metrics."""


def all_online_metrics():
  """Get all the online metrics."""
  return _OnlineMetric.public_subclasses()


class _DispersionAcrossRuns(_OnlineMetric):
  """Computes dispersion across runs."""

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATP'
  bigger_is_better = False

  def __init__(self,
               dispersion_fn,
               lowpass_thresh=None,
               eval_points=None,
               window_size=None,
               baseline=None):
    """Initializes parameters for computing dispersion across runs.

    Args:
      dispersion_fn: Function for computing dispersion.
      lowpass_thresh: Frequency threshold for low-pass filtering. This is the
        point at which the gain drops to 1/sqrt(2) that of the passband (the
        "-3 dB point"). The threshold should be normalized between 0 and 1,
        where 1 is the Nyquist frequency, pi radians/sample. See documentation
        for scipy.signal.butter.
      eval_points: A list or Numpy array of length [# timepoints]. Standard
        deviation will be computed at these timepoints. Set to None to select
        all valid eval points.
      window_size: If not None, defines a window centered at each eval point. We
        evaluate dispersion across runs at the timepoints closest to each eval
        point (but still within each window). This is useful when the available
        timepoints are not precisely aligned across runs. If None, we evaluate
        exactly at each eval point.
      baseline: Set to "curve_range" to normalize by the curve range, defined as
        the 95th percentile minus the start value. Set to a float to simply
        divide by that value. Set to None for no normalization.
    """
    self._dispersion_fn = dispersion_fn
    self.lowpass_thresh = lowpass_thresh
    self.eval_points = eval_points
    self.window_size = window_size
    self.baseline = baseline

  def __call__(self, curves):
    """Computes normalized dispersion across runs.

    Args:
      curves: A list of learning curves, each a 2D numpy array where curve[0, :]
        is the timepoint variable and curve[1, :] is the dependent variable.

    Returns:
       Dispersion across runs, computed at each of the eval_points.
       (Numpy array with length n_eval_points).
    """
    utils.assert_non_empty(curves)

    # perform preprocessing for across-runs metrics
    eval_point_values = utils.across_runs_preprocess(
        curves, self.eval_points, self.window_size, self.lowpass_thresh)

    # compute dispersion across curves
    result = self._dispersion_fn(eval_point_values)

    if self.baseline == 'curve_range':
      curve_ranges = utils.curve_range(curves)
      result /= np.median(curve_ranges)
    elif self.baseline:
      result = result / self.baseline

    return result


@gin.configurable
class IqrAcrossRuns(_DispersionAcrossRuns):
  """Computes interquartile range across runs."""

  def __init__(self,
               lowpass_thresh=None,
               eval_points=None,
               window_size=None,
               baseline=None):
    super(IqrAcrossRuns, self).__init__(
        dispersion_fn=lambda x: scipy.stats.iqr(x, axis=0),
        lowpass_thresh=lowpass_thresh,
        eval_points=eval_points,
        window_size=window_size,
        baseline=baseline)


@gin.configurable
class MadAcrossRuns(_DispersionAcrossRuns):
  """Computes median absolute deviation across runs."""

  def __init__(self,
               lowpass_thresh=None,
               eval_points=None,
               window_size=None,
               baseline=None):
    super(MadAcrossRuns, self).__init__(
        dispersion_fn=lambda x: utils.median_absolute_deviations(x, axis=0),
        lowpass_thresh=lowpass_thresh,
        eval_points=eval_points,
        window_size=window_size,
        baseline=baseline)


@gin.configurable
class StddevAcrossRuns(_DispersionAcrossRuns):
  """Computes standard deviation across runs."""

  def __init__(self,
               lowpass_thresh=None,
               eval_points=None,
               window_size=None,
               baseline=None):
    super(StddevAcrossRuns, self).__init__(
        dispersion_fn=lambda x: np.std(x, axis=0, ddof=1),
        lowpass_thresh=lowpass_thresh,
        eval_points=eval_points,
        window_size=window_size,
        baseline=baseline)


class _DispersionWithinRuns(_OnlineMetric):
  """Computes dispersion within runs."""

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATRP'
  bigger_is_better = False

  def __init__(self,
               dispersion_fn,
               window_size=None,
               eval_points=None,
               baseline=None,
               detrend=True):
    """Initializes parameters for computing dispersion within runs.

    Args:
      dispersion_fn: Function for computing dispersion.
      window_size: The number of timepoints in the window. Set to None to use
        window_size = entire length of the run.
      eval_points: A list or Numpy array of length [# timepoints]. Standard
        deviation will be computed at these timepoints. Set to None to use all
        valid timepoints (given the window_size).
      baseline: Set to "curve_range" to normalize by the curve range, defined as
        the 95th percentile minus the start value. Set to a float to simply
        divide by that value. Set to None for no normalization.
      detrend: If True, detrend by differencing, before computing dispersion.
    """
    self._dispersion_fn = dispersion_fn
    self.window_size = window_size
    self.eval_points = eval_points
    self.baseline = baseline
    self.detrend = detrend

  def __call__(self, curves):
    """Computes dispersion within runs.

    Args:
      curves: A list of learning curves, each a 2D numpy array where curve[0, :]
        is the timepoint variable and curve[1, :] is the dependent variable.

    Returns:
       Dispersion within runs, computed at each eval_point for each run.
       (Numpy array with size n_run x n_eval_points.)
    """
    utils.assert_non_empty(curves)

    # Detrend by differencing.
    if self.detrend:
      diff_curves = utils.differences(curves)
    else:
      diff_curves = curves

    dispersions = []
    # Process each curve separately, because length of run may differ for each.
    for curve, diff_curve in zip(curves, diff_curves):
      eval_points = copy.deepcopy(self.eval_points)
      window_size = copy.deepcopy(self.window_size)

      # Determine eval_points and window_size, if needed (based on diff_curve).
      if self.eval_points is None or self.window_size is None:
        if self.window_size is None:
          valid_eval_points = utils.get_all_valid_eval_points([diff_curve], 1)
          window_size = valid_eval_points.max() - valid_eval_points.min() + 1
        if self.eval_points is None:
          eval_points = utils.get_all_valid_eval_points([diff_curve],
                                                        window_size)

      # Compute dispersion for the curve.
      diffcurve_dispers = utils.apply_window_fn(
          [diff_curve], eval_points, self._dispersion_fn, window_size)

      if self.baseline == 'curve_range':
        curve_range = utils.curve_range([curve])[0]
        diffcurve_dispers = diffcurve_dispers / curve_range
      elif self.baseline:
        diffcurve_dispers /= self.baseline
      dispersions.extend(diffcurve_dispers)

    return np.array(dispersions)


@gin.configurable
class StddevWithinRuns(_DispersionWithinRuns):
  """Computes standard deviation within runs."""

  def __init__(self, window_size=None, eval_points=None, baseline=None):
    super(StddevWithinRuns, self).__init__(functools.partial(np.std, ddof=1),
                                           window_size,
                                           eval_points,
                                           baseline,
                                           True)


@gin.configurable
class IqrWithinRuns(_DispersionWithinRuns):
  """Computes inter-quartile range within runs."""

  def __init__(self, window_size=None, eval_points=None, baseline=None):
    super(IqrWithinRuns, self).__init__(scipy.stats.iqr,
                                        window_size,
                                        eval_points,
                                        baseline,
                                        True)


@gin.configurable
class MadWithinRuns(_DispersionWithinRuns):
  """Computes median absolute deviation within runs."""

  def __init__(self, window_size=None, eval_points=None, baseline=None):
    super(MadWithinRuns, self).__init__(utils.median_absolute_deviations,
                                        window_size,
                                        eval_points,
                                        baseline,
                                        True)


@gin.configurable
class MaxDrawdown(_OnlineMetric):
  """Maximum drawdown (borrowed from economics/finance).

  Maximum drawdown measures the largest peak-to-valley loss on each curve.

  https://en.wikipedia.org/wiki/Drawdown_(economics)
  """

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = False

  def __init__(self, baseline=None, mean_normalize=False):
    """Initializes parameters for computing maximum drawdown.

    Args:
      baseline: If not None, this is a float value that is subtracted from the
        curves as the first step of pre-processing.
      mean_normalize: If True, normalize curves by the mean value of each curve
        during preprocessing (after subtracting baseline, if available).
    """
    self.baseline = baseline
    self.mean_normalize = mean_normalize

  def __call__(self, curves):
    """Compute maximum drawdown."""
    utils.assert_non_empty(curves)

    if self.baseline is not None:
      curves = utils.subtract_baseline(curves, self.baseline)
    if self.mean_normalize:
      curves = utils.mean_normalization(curves)

    mdd = np.empty(len(curves))
    for i, curve in enumerate(curves):
      dependent_vals = curve[1, :]
      drawdown = utils.compute_drawdown(dependent_vals)
      mdd[i] = np.max(drawdown)
    return mdd


@gin.configurable
class HighFreqEnergyWithinRuns(_OnlineMetric):
  """Computes the energy of the signal above a given frequency threshold.

  Normalized by the total energy of the signal.
  This is a measure of dispersion within runs.
  """

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = False

  def __init__(self, thresh):
    """Initialize parameters.

    Args:
      thresh: frequency threshold
    """
    self.thresh = thresh

  def __call__(self, curves):
    """Computes energy of the signal above a given frequency threshold.

    Normalized by the total energy of the signal.

    Args:
      curves: A list of learning curves, each a 2D numpy array where curve[0, :]
        is the timepoint variable and curve[1, :] is the dependent variable.

    Returns:
      Amount of energy above the given frequency threshold, normalized by the
      total energy of the signal.
    """
    utils.assert_non_empty(curves)

    energies = []
    for curve in curves:
      data = curve[1, :]
      power_spectrum = np.abs(np.fft.fft(data))**2
      time_step = curve[0, 1] - curve[0, 0]
      # TODO(scychan) above assumes equal spacing
      freqs = np.fft.fftfreq(data.size, time_step)
      energy_above_thresh = (np.sum(power_spectrum[freqs > self.thresh]) /
                             np.sum(power_spectrum[freqs > 0]))
      energies.append(energy_above_thresh)

    return energies


class _CVaR(_OnlineMetric):
  """Computes conditional value at risk (CVaR), aka "expected shortfall".

  For each learning curve, this metric takes the expected value on the curve
  values that fall below the quantile defined by `alpha` (if computing on the
  lower tail), or above the quantile defined by 1 - `alpha` (if computing on
  the upper tail).

  https://en.wikipedia.org/wiki/Expected_shortfall
  """

  def __init__(self,
               target,
               tail,
               alpha=0.05,
               baseline=None,
               lowpass_thresh=None,
               eval_points=None,
               window_size=None):
    """Initializes parameters for computing CVaR.

    Args:
      target: What data to perform CVaR on. Options:
        'across' - across the training runs, evaluated at the eval points after
          low-pass thresholding.
        'diffs' - the timepoint to timepoint differences (per training curve)
        'raw' - the raw values (per training curve)
        'drawdown' - drawdown (per training curve)
                     https://en.wikipedia.org/wiki/Drawdown_(economics)
      tail: 'lower' or 'upper' tail
      alpha: The "value at risk" (VaR) cutoff point, a float in the range [0,1].
        To compute CVaR we computed expected value below this quantile.
      baseline: Set to "curve_range" to normalize by the curve range, defined as
        the 95th percentile minus the start value. Set to a float to simply
        divide by that value. Set to None for no normalization.
      lowpass_thresh: [for target == 'across' only] The frequency threshold for
        low-pass thresholding before computing CVaR
      eval_points: [for target == 'across' only] A list or Numpy array of length
        [# timepoints]. CVaR will be computed at these timepoints. Set to None
        to select all valid eval points.
      window_size: [For target == 'across' only]. If not None, defines a window
        centered at each eval point. We evaluate CVaR across runs at the
        timepoints closest to each eval point (but still within each window).
        This is useful when the available timepoints are not precisely aligned
        across runs. If None, we evaluate exactly at each eval point.
    """
    if target not in ['across', 'diffs', 'raw', 'drawdown']:
      raise ValueError("target must be 'across', 'diffs', 'raw', or "
                       "'drawdown'.")

    self.target = target
    self.tail = tail
    self.alpha = alpha
    self.baseline = baseline
    self.lowpass_thresh = lowpass_thresh
    self.eval_points = eval_points
    self.window_size = window_size

  def __call__(self, curves):
    """Computes CVaR for a list of curves.

    Args:
      curves: A list of learning curves, each a 2D numpy array where curve[0, :]
        is the timepoint variable and curve[1, :] is the dependent variable.

    Returns:
      for self.target in ['diffs', 'raw', 'drawdown']:
         A 1-D numpy array of CVaR values, one per curve in the input
         (length = the number of curves in the input).
      for self.target == 'across':
        A 1-D numpy array of CVaR values, one per eval point
        (length = number of eval points)
    """
    utils.assert_non_empty(curves)

    if self.baseline == 'curve_range':
      curve_ranges = utils.curve_range(curves)
      curves = utils.divide_by_baseline(curves, curve_ranges)
    elif self.baseline:
      curves = utils.divide_by_baseline(curves, self.baseline)

    cvar_list = []
    if self.target == 'across':
      # Compute CVaR across curves (at each eval point)
      eval_point_vals = utils.across_runs_preprocess(curves, self.eval_points,
                                                     self.window_size,
                                                     self.lowpass_thresh)
      n_eval_points = eval_point_vals.shape[1]
      for i_point in range(n_eval_points):
        cvar = utils.compute_cvar(eval_point_vals[:, i_point], self.tail,
                                  self.alpha)
        cvar_list.append(cvar)
    else:
      # Compute CVaR within curves (one per curve).
      for curve in curves:
        dependent_var = curve[1, :]
        if self.target == 'raw':
          pass
        elif self.target == 'diffs':
          normalized_diffs = utils.differences([curve])[0]
          dependent_var = normalized_diffs[1, :]
        elif self.target == 'drawdown':
          dependent_var = utils.compute_drawdown(dependent_var)

        cvar = utils.compute_cvar(dependent_var, self.tail, self.alpha)
        cvar_list.append(cvar)

    return np.array(cvar_list)


@gin.configurable
class LowerCVaROnDiffs(_CVaR):

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = True

  def __init__(self, alpha=0.05, baseline=None):
    super(LowerCVaROnDiffs, self).__init__(
        target='diffs',
        tail='lower',
        alpha=alpha,
        baseline=baseline,
        lowpass_thresh=None,
        eval_points=None)


@gin.configurable
class UpperCVaROnDiffs(_CVaR):

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = True

  def __init__(self, alpha=0.05, baseline=None):
    super(UpperCVaROnDiffs, self).__init__(
        target='diffs',
        tail='upper',
        alpha=alpha,
        baseline=baseline,
        lowpass_thresh=None,
        eval_points=None)


@gin.configurable
class LowerCVaROnRaw(_CVaR):

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = True

  def __init__(self, alpha=0.05, baseline=None):
    super(LowerCVaROnRaw, self).__init__(
        target='raw',
        tail='lower',
        alpha=alpha,
        baseline=baseline,
        lowpass_thresh=None,
        eval_points=None)


@gin.configurable
class UpperCVaROnRaw(_CVaR):

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = True

  def __init__(self, alpha=0.05, baseline=None):
    super(UpperCVaROnRaw, self).__init__(
        target='raw',
        tail='upper',
        alpha=alpha,
        baseline=baseline,
        lowpass_thresh=None,
        eval_points=None)


@gin.configurable
class LowerCVaROnDrawdown(_CVaR):

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = False

  def __init__(self, alpha=0.05, baseline=None):
    super(LowerCVaROnDrawdown, self).__init__(
        target='drawdown',
        tail='lower',
        alpha=alpha,
        baseline=baseline,
        lowpass_thresh=None,
        eval_points=None)


@gin.configurable
class UpperCVaROnDrawdown(_CVaR):

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = False

  def __init__(self, alpha=0.05, baseline=None):
    super(UpperCVaROnDrawdown, self).__init__(
        target='drawdown',
        tail='upper',
        alpha=alpha,
        baseline=baseline,
        lowpass_thresh=None,
        eval_points=None)


@gin.configurable
class LowerCVaROnAcross(_CVaR):
  """Lower CVaR across training runs."""

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATP'
  bigger_is_better = True

  def __init__(self,
               alpha=0.05,
               baseline=None,
               lowpass_thresh=None,
               eval_points=None,
               window_size=None):
    super(LowerCVaROnAcross, self).__init__(
        target='across',
        tail='lower',
        alpha=alpha,
        baseline=baseline,
        lowpass_thresh=lowpass_thresh,
        eval_points=eval_points,
        window_size=window_size)


@gin.configurable
class UpperCVaROnAcross(_CVaR):
  """Upper CVaR across training runs."""

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATP'
  bigger_is_better = True

  def __init__(self,
               alpha=0.05,
               baseline=None,
               lowpass_thresh=None,
               eval_points=None,
               window_size=None):
    super(UpperCVaROnAcross, self).__init__(
        target='across',
        tail='upper',
        alpha=alpha,
        baseline=baseline,
        lowpass_thresh=lowpass_thresh,
        eval_points=eval_points,
        window_size=window_size)


@gin.configurable
class MedianPerfDuringTraining(_OnlineMetric):
  """Median performance, within windows at specified points in training."""

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATRP'
  bigger_is_better = True

  def __init__(self, window_size=None, eval_points=None, baseline=None):
    """Initializes parameters for computing median performance.

    Args:
      window_size: The number of timepoints in the window. Set to None to use
        window_size = entire length of the run.
      eval_points: A list or Numpy array of length [# timepoints]. Performance
        will be computed at these timepoints. Set to None to use all valid
        timepoints (given the window_size).
      baseline: If this is a single float, we normalize using
        normalized = perf / baseline. If this is a tuple of floats (low, high),
        we normalize using normalized = (perf - low) / (high - low). If None or
        if an iterable that contains None, we do not perform any normalization.
    """
    self.window_size = window_size
    self.eval_points = eval_points
    self.baseline = baseline

  def __call__(self, curves):
    """Computes median performance.

    Args:
      curves: A list of learning curves, each a 2D numpy array where curve[0, :]
        is the timepoint variable and curve[1, :] is the dependent variable.

    Returns:
       Median performance, computed in a window at each eval_point for each run.
       (Numpy array with size n_run x n_eval_points.)
    """
    utils.assert_non_empty(curves)

    # Determine eval_points and window_size, if needed.
    eval_points = copy.deepcopy(self.eval_points)
    window_size = copy.deepcopy(self.window_size)
    if eval_points is None or window_size is None:
      if window_size is None:
        valid_eval_points = utils.get_all_valid_eval_points(curves, 1)
        window_size = valid_eval_points.max() - valid_eval_points.min() + 1
      if eval_points is None:
        eval_points = utils.get_all_valid_eval_points(curves, window_size)

    curves = self._normalize(curves)

    perf = utils.apply_window_fn(curves, eval_points, np.median, window_size)
    return perf

  def _normalize(self, curves):
    """Normalize curves depending on setting of self.baseline."""
    if self.baseline is None:
      return curves

    if isinstance(self.baseline, tuple):
      if None in self.baseline:
        return curves
      if len(self.baseline) != 2:
        raise ValueError('If baseline is a tuple it must be of the form '
                         '(low, high). Got %r' % self.baseline)
      low, high = self.baseline
    else:
      low = 0
      high = self.baseline
    return utils.band_normalization(curves, low, high)


# Maintain a registry linking metric names to classes.
REGISTRY = {
    metric.__name__: metric for metric in all_online_metrics()
}
