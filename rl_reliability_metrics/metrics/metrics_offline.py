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

"""Offline metrics for evaluating robustness of an RL algorithm.

Given the performance of an algorithm on a set of rollouts, these metrics
provide measures of the robustness of the RL algorithm.
"""

import abc
import functools
import gin
import numpy as np
from rl_reliability_metrics.metrics import metric_utils as utils
from rl_reliability_metrics.metrics import metrics_base
import scipy.stats
import six


@six.add_metaclass(abc.ABCMeta)
class _OfflineMetric(metrics_base.Metric):
  """Base class for offline metrics."""


def all_offline_metrics():
  """Get all the offline metrics."""
  return _OfflineMetric.public_subclasses()


class _DispersionAcrossRollouts(_OfflineMetric):
  """Computes dispersion across rollouts of a fixed policy.

  A rollout may be a fixed number of actions, or an episode otherwise defined.
  """

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = False

  def __init__(self, dispersion_fn, baseline=None):
    """Initializes parameters.

    Args:
      dispersion_fn: Function for computing dispersion.
      baseline: Set to "median_perf" to normalize by the median performance
        across rollouts (within each rollout set). Set to a float to normalize
        by that value. Set to None for no normalization.
    """
    self._dispersion_fn = dispersion_fn
    self.baseline = baseline

  def __call__(self, rollout_sets):
    """Computes dispersion across rollouts.

    Args:
      rollout_sets: A list of rollout sets, with length n_rollout_sets.
        Each element of the list corresponds to the performance values of one
        set of rollouts that we will measure dispersion across (e.g. for a
        single model checkpoint). It is a 2D numpy array where rollouts[0, :] is
        just an index variable (e.g. range(0, n_rollouts)) and rollouts[1, :]
        are the performances per rollout.

    Returns:
       Dispersion across rollouts, computed for each rollout set.
       (1-D Numpy array with length = n_rollout_sets)
    """
    utils.assert_non_empty(rollout_sets)

    dispersions = []
    for rollout_set in rollout_sets:
      dispersion = self._dispersion_fn(rollout_set[1, :])
      dispersions.append(dispersion)

    dispersions = np.array(dispersions)

    if self.baseline:
      if self.baseline == 'median_perf':
        divisor = utils.median_rollout_performance(rollout_sets)
      else:
        divisor = self.baseline
      dispersions /= divisor
    return dispersions


@gin.configurable
class MadAcrossRollouts(_DispersionAcrossRollouts):
  """Computes median absolute deviation across rollouts of a fixed policy.

  A rollout may be a fixed number of actions, or an episode otherwise defined.
  """

  def __init__(self, baseline=None):
    super(MadAcrossRollouts, self).__init__(
        utils.median_absolute_deviations,
        baseline)


@gin.configurable
class IqrAcrossRollouts(_DispersionAcrossRollouts):
  """Computes inter-quartile range across rollouts of a fixed policy.

  A rollout may be a fixed number of actions, or an episode otherwise defined.
  """

  def __init__(self, baseline=None):
    super(IqrAcrossRollouts, self).__init__(scipy.stats.iqr, baseline)


@gin.configurable
class StddevAcrossRollouts(_DispersionAcrossRollouts):
  """Computes median absolute deviation across rollouts of a fixed policy.

  A rollout may be a fixed number of actions, or an episode otherwise defined.
  """

  def __init__(self, baseline=None):
    super(StddevAcrossRollouts, self).__init__(
        functools.partial(np.std, ddof=1), baseline)


class _CVaRAcrossRollouts(_OfflineMetric):
  """Computes CVaR (as a measure of risk) across rollouts of a fixed policy.

  A rollout may be a fixed number of actions, or an episode otherwise defined.
  """

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = True

  def __init__(self, tail, alpha=0.05, baseline=None):
    """Initializes parameters for computing CVaR across rollouts.

    Args:
      tail: Set to "lower" or "upper" accordingly to compute CVaR on the lower
        or upper tail of the distribution.
      alpha: The threshold for computing CVaR. If tail="lower", we compute on
        the part of the distribution <= the (alpha)-quantile. If tail="upper",
        we compute on the part of the distribution >=  the (1-alpha)-quantile.
      baseline: A float value. When set, the rollout data will be divided by
        this baseline before we compute CVaR.
    """
    self.tail = tail
    self.alpha = alpha
    self.baseline = baseline

  def __call__(self, rollout_sets):
    """Computes CVaR across rollouts of a fixed policy.

    Args:
      rollout_sets: A list of rollout sets, with length n_rollout_sets.
        Each element of the list corresponds to the performance values of one
        set of rollouts that we will measure dispersion across (e.g. for a
        single model checkpoint). It is a 2D numpy array where rollouts[0, :] is
        just an index variable (e.g. range(0, n_rollouts)) and rollouts[1, :]
        are the performances per rollout.

    Returns:
      CVaR across rollouts, computed for each rollout set.
       (1-D Numpy array with length = n_rollout_sets)
    """
    utils.assert_non_empty(rollout_sets)

    if self.baseline is not None:
      if self.baseline == 'median_perf':
        divisor = utils.median_rollout_performance(rollout_sets)
      else:
        divisor = self.baseline
      rollout_sets = utils.divide_by_baseline(rollout_sets, divisor)

    cvar_list = []
    # Compute CVaR within each rollout set.
    for rollout_set in rollout_sets:
      dependent_var = rollout_set[1, :]
      cvar = utils.compute_cvar(dependent_var, self.tail, self.alpha)
      cvar_list.append(cvar)

    return np.array(cvar_list)


@gin.configurable
class LowerCVaRAcrossRollouts(_CVaRAcrossRollouts):

  def __init__(self, alpha=0.05, baseline=None):
    super(LowerCVaRAcrossRollouts, self).__init__('lower', alpha, baseline)


@gin.configurable
class UpperCVaRAcrossRollouts(_CVaRAcrossRollouts):

  def __init__(self, alpha=0.05, baseline=None):
    super(UpperCVaRAcrossRollouts, self).__init__('upper', alpha, baseline)


@gin.configurable
class MedianPerfAcrossRollouts(_OfflineMetric):
  """Median performance for each rollout set."""

  # Set metric properties (see metrics_base.Metric).
  result_dimensions = 'ATR'
  bigger_is_better = True

  def __init__(self, baseline=None):
    """Initializes parameters for computing median performance.

    Args:
      baseline: If this is a single float, we normalize using
        normalized = perf / baseline. If this is a tuple of floats (low, high),
        we normalize using normalized = (perf - low) / (high - low). If None or
        if an iterable that contains None, we do not perform any normalization.
    """
    self.baseline = baseline

  def __call__(self, rollout_sets):
    """Computes median performance for each rollout set.

    Args:
      rollout_sets: A list of rollout sets, with length n_rollout_sets.
        Each element of the list corresponds to the performance values of one
        set of rollouts that we will measure median performance for (e.g. for a
        single model checkpoint). It is a 2D numpy array where rollout_set[0, :]
        is just an index variable (e.g. range(0, n_rollouts)) and
        rollout_set[1, :] are the performances per rollout.

    Returns:
      Median performance for each rollout set.
      (1-D Numpy array with length = n_rollout_sets)
    """
    rollout_sets = self._normalize(rollout_sets)
    perf = [np.median(rollout_set[1, :]) for rollout_set in rollout_sets]
    return perf

  def _normalize(self, rollout_sets):
    """Normalize curves depending on setting of self.baseline."""
    if self.baseline is None:
      return rollout_sets

    if isinstance(self.baseline, tuple):
      if None in self.baseline:  # E.g., (None, None) or (None, some float)
        return rollout_sets
      if len(self.baseline) != 2:
        raise ValueError('If baseline is a tuple it must be of the form '
                         '(low, high). Got %r' % self.baseline)
      low, high = self.baseline
    else:
      low = 0
      high = self.baseline
    return utils.band_normalization(rollout_sets, low, high)


# Maintain a registry linking metric names to classes.
REGISTRY = {
    metric.__name__: metric for metric in all_offline_metrics()
}
