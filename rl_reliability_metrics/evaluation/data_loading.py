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

"""Utilities for loading data."""

import csv

import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import event_accumulator


class DataLoader:
  """Class for loading CSV- or Tensorboard-formatted input data."""

  def __init__(self,
               dependent_variable='Metrics/AverageReturn',
               timepoint_variable=None,
               discard_tails=True,
               align_on_global_step=True):
    """Initialize class for loading CSV- or Tensorboard-formatted input data.

    Args:
      dependent_variable: Name of CSV column or Tensorboard summary variable
        that should be loaded for analysis of reliability. If provided with a
        list of names, we try each one in sequence until the first one that
        works.
      timepoint_variable: Name of CSV column or Tensorboard summary variable
        that defines a "timepoint" (i.e. the independent variable). If provided
        with a list of names, we try each one in sequence until the first one
        that works. For Tensorboard data, this can be set to None, in which case
        the timepoint variable is simply loaded from the 'steps' value of the
        dependent_variable summaries.
      discard_tails: If True, discard extra "tails" that are generated when
        there is a restart.
      align_on_global_step: [Only applicable for Tensorboard data.] If True, we
        assume that the 'step' fields of both the timepoint_variable and the
        dependent_variable refer to a global step that can be used to align
        dependent_variable with timepoint_variable. If dependent_variable and
        timepoint_variable have different sets of step values, we only load
        those steps that exist in common for both variables. If False, we assume
        that the 'step' field of the dependent_variable refers to the actual
        values of the timepoint_variable, and the two variables are aligned on
        that instead. Whichever variable is used for alignment (global step or
        timepoints) are also used to determine where restarts occurred, so that
        any loose "tails" can be discarded if needed. This variable is
        irrelevant if timepoint_variable == None.
    """
    self.dependent_variable = dependent_variable
    self.timepoint_variable = timepoint_variable
    self.discard_tails = discard_tails
    self.align_on_global_step = align_on_global_step

  def load_input_data(self, run_paths):
    """Load input data into a list of numpy arrays.

    Args:
      run_paths: List of paths containing data for all runs of an experiment.
        See load_csv_data and load_tensorboard_data for requirements on how the
        data should be formatted.

    Returns:
      A list of numpy arrays.

      If the input data are training curves, each numpy array represents a curve
      and has dims [2 x curve_length]. curve[0, :] are the timepoints and
      curve[1, :] are the dependent variable. Each curve is sorted by the
      timepoint variable.

      If the input data are sets of rollouts, each numpy array represents a
      single set of rollouts (e.g. n_rollouts for a single model checkpoint) and
      has dims [2 x n_rollouts]. rollouts[0, :] is just an index variable (e.g.
      range(0, n_rollouts)) and rollouts[1, :] are the performances per rollout.
      Each rollout set is sorted by the index variable.
    """
    if '.csv' in run_paths[0]:
      return self.load_csv_data(run_paths)
    else:
      return self.load_tensorboard_data(run_paths)

  def load_csv_data(self, run_paths):
    """Loads CSV-formatted input data, into a list of numpy arrays.

    Input data may be a set of training curves for each run, or the performance
    values for sets of rollouts.

    Args:
      run_paths: List of paths to .csv files for all runs of an experiment.
        This should be a list of CSV filepaths, one file per run. One column
        should define a "timepoint" (the 'timepoint_variable'), and one column
        should contain the variable to be analyzed (the 'dependent_variable').
        The name of each column should be in the first row.

    Returns:
      A list of numpy arrays.

      If the input data are training curves, each numpy array represents a curve
      and has dims [2 x curve_length]. curve[0, :] are the timepoints and
      curve[1, :] are the dependent variable. Each curve is sorted by the
      timepoint variable.

      If the input data are sets of rollouts, each numpy array represents a
      single set of rollouts (e.g. n_rollouts for a single model checkpoint) and
      has dims [2 x n_rollouts]. rollouts[0, :] is just an index variable (e.g.
      range(0, n_rollouts)) and rollouts[1, :] are the performances per rollout.
      Each rollout set is sorted by the index variable.
    """

    def _find_column_index(target_names, header):
      """For the first value of target_names found in header, return the index in header."""
      if not isinstance(target_names, (list, tuple)):
        target_names = [target_names]
      col_index = None
      for target_name in target_names:
        try:
          col_index = header.index(target_name)
          break
        except ValueError:
          continue
      if col_index is None:
        raise ValueError('None of the names (%s) found in CSV header (%s)' %
                         (target_names, header))
      return col_index

    curves = []
    for csv_path in run_paths:
      # Determine which columns correspond to timepoint and dependent variables.
      with open(csv_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader)
      timepoint_col = _find_column_index(self.timepoint_variable, header)
      dependent_col = _find_column_index(self.dependent_variable, header)

      # Load the data
      all_data = np.loadtxt(csv_path, skiprows=1, delimiter=',')
      steps = all_data[:, timepoint_col]
      values = all_data[:, dependent_col]

      # Discard tails from restarts.
      if self.discard_tails:
        steps, values = self._discard_tails_from_restarts(steps, values)

      curve = np.array([steps, values])
      curves.append(curve)
    return curves

  def load_tensorboard_data(self, run_paths):
    """Loads Tensorboard input data, into a list of numpy arrays.

    Input data may be a set of training curves for each run, or the performance
    values for sets of rollouts.

    Args:
      run_paths: List of paths to directories containing Tensorboard summaries
        for all the runs of an experiment, one directory per run. Summaries must
        include a scalar or tensor summary that defines the variable to be
        analyzed (the 'dependent_variable'). Optionally they may also have a
        scalar or tensor summary that defines a "timepoint" (the
        'timepoint_variable').

    Returns:
      A list of numpy arrays.

      If the input data are training curves, each numpy array represents a curve
      and has dims [2 x curve_length]. curve[0, :] are the timepoints and
      curve[1, :] are the dependent variable. Each curve is sorted by the
      timepoint variable.

      If the input data are sets of rollouts, each numpy array represents a
      single set of rollouts (e.g. n_rollouts for a single model checkpoint) and
      has dims [2 x n_rollouts]. rollouts[0, :] is just an index variable (e.g.
      range(0, n_rollouts)) and rollouts[1, :] are the performances per rollout.
      Each rollout set is sorted by the index variable.
    """

    if self.align_on_global_step:
      restart_determiner_x = 'step'
    else:
      restart_determiner_x = 'value'
    restart_determiner_y = 'step'

    curves = []
    for run_dir in run_paths:
      accumulator = event_accumulator.EventAccumulator(
          run_dir,
          size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE)
      accumulator.Reload()

      # Load the dependent variable.
      y_vals, y_steps = self._extract_summary(accumulator,
                                              self.dependent_variable,
                                              restart_determiner_y)
      y_vals_dict = {step: val for step, val in zip(y_steps, y_vals)}

      # Load the timepoint variable.
      if self.timepoint_variable is None:
        # Load from the step values of y.
        steps_to_load = set(y_steps)
        x_steps = y_steps
        x_vals_dict = {step: step for step in x_steps}
      else:
        # Load from summaries.
        x_vals, x_steps = self._extract_summary(accumulator,
                                                self.timepoint_variable,
                                                restart_determiner_x)
        if self.align_on_global_step:
          x_vals_dict = {step: val for step, val in zip(x_steps, x_vals)}
        else:
          x_vals_dict = {val: val for val in x_vals}

        # Find steps in common between x and y
        if self.align_on_global_step:
          steps_to_load = set(x_steps).intersection(y_steps)
        else:
          steps_to_load = set(x_vals).intersection(y_steps)

      # Save x- and y-values into a 2-D numpy array [2 x n_timepoints].
      # curve[0, :] are the timepoints.
      # curve[1, :] are the dependent variable.
      # Only keep the values of x and y that have steps in common.
      steps_to_load_list = list(steps_to_load)
      curve = np.array([[x_vals_dict[key] for key in steps_to_load_list],
                        [y_vals_dict[key] for key in steps_to_load_list]])

      # Order the curve according to the values of timepoint_variable.
      ordering_by_timepoint = curve[0, :].argsort()
      curve = curve[:, ordering_by_timepoint]

      curves.append(curve)
    return curves

  def _extract_summary(self,
                       accumulator,
                       summary_name,
                       restart_determiner=None):
    """Extracts summary steps and values for a given accumulator.

    Summaries may be saved as TensorEvent (TF v2) or ScalarEvent (TF v1).

    Args:
      accumulator: A Tensorboard EventAccumulator object.
      summary_name: The name ("tag") of the summary that should be extracted. If
        given as a list of strings rather than a single string, we try each one
        until the first tag that succeeds.
      restart_determiner: See "determiner" arg of discard_tails_from_restarts.

    Returns:
      (event values, event steps)
    """
    if not isinstance(summary_name, (list, tuple)):
      summary_name = [summary_name]

    # Check if we are loading from Tensors or Scalars
    tensor_keys = accumulator.tensors.Keys()
    load_tensors = bool(set(tensor_keys).intersection(summary_name))

    # Load summaries -- try each key in the list until the first one that works.
    summaries = None
    for name in summary_name:
      try:
        if load_tensors:
          summaries = accumulator.Tensors(name)
        else:
          summaries = accumulator.Scalars(name)
      except KeyError:
        continue
      else:
        break
    assert summaries  # Assert that summaries were actually loaded.

    # Load steps and values.
    steps = [summary.step for summary in summaries]
    values = [
        self._get_summary_value(summary, load_tensors) for summary in summaries
    ]

    # Discard the "tails" from restarts.
    if self.discard_tails:
      steps, values = self._discard_tails_from_restarts(steps, values,
                                                        restart_determiner)

    return values, steps

  def _get_summary_value(self, summary, load_tensors):
    if load_tensors:
      proto = summary.tensor_proto
      value = tf.make_ndarray(proto)
      return value.item()
    else:
      return summary.value

  def _discard_tails_from_restarts(self, steps, values, determiner='step'):
    """Discard extra "tails" that are generated when there is a restart.

    Args:
      steps: List of global_steps loaded from summaries.
      values: List of values loaded from summaries.
      determiner: If 'step', determine restarts based on the steps. If 'value',
        determine restarts based on the values. If None, do not discard any
        values.

    Returns:
      new_steps: List of global_steps with tails discarded.
      new_values: List of values with tails discarded.
    """
    if determiner is None:
      # Do not discard tails.
      return steps, values

    # Find the restarts.
    if determiner == 'step':
      array_to_analyze = steps
    elif determiner == 'value':
      array_to_analyze = values
    else:
      raise ValueError('Invalid value for determiner: %s' % determiner)
    array_to_analyze = np.array(array_to_analyze)
    restarts = np.where(np.diff(array_to_analyze) <= 0)[0]

    if not restarts.size:
      # No restarts.
      return steps, values

    restart_x = array_to_analyze[restarts + 1]

    # Initialize with first restart.
    restart_ind = 0
    restart_t = restarts[restart_ind]
    restart_xval = restart_x[restart_ind]

    # Rebuild the data, excluding "tails" before restarts.
    new_steps, new_values = [], []
    for t, (step, value) in enumerate(zip(steps, values)):
      to_compare = step if determiner == 'step' else value
      if to_compare < restart_xval:
        # We are not in a tail, ok to append.
        new_steps.append(step)
        new_values.append(value)
      if t >= restart_t:
        # Go to the next restart value.
        if restart_ind < len(restarts) - 1:
          restart_ind += 1
          restart_t = restarts[restart_ind]
          restart_xval = restart_x[restart_ind]
        else:
          restart_xval = np.inf

    return new_steps, new_values
