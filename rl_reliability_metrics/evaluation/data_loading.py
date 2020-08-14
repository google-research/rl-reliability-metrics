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

import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import event_accumulator


def load_input_data(run_dirs, dependent_variable, timepoint_variable,
                    align_on_global_step):
  """Loads input data, as a list of numpy arrays.

  Input data may be a set of training curves for each run, or the performance
  values for sets of rollouts.

  If there are restarts during training, we discard the loose "tails" that were
  generated.

  Args:
    run_dirs: List of paths to directories containing Tensorboard summaries for
      all the runs of an experiment, one directory per run. Summaries must
      include a scalar or tensor summary that defines the variable to be
      analyzed (the 'dependent_variable'). Optionally they may also have a
      scalar or tensor summary that defines a "timepoint" (the
      'timepoint_variable').
    dependent_variable: Name of Tensorboard summary that should be loaded for
      analysis of robustness. If provided with a list of names, we try each one
      in sequence until the first one that works.
    timepoint_variable: Name of Tensorboard summary that defines a "timepoint"
      (i.e. the independent variable). Set None to simply use the 'steps' value
      of the dependent_variable summaries as the timepoint variable. If provided
      with a list of names, we try each one in sequence until the first one that
      works.
    align_on_global_step: If True, we assume that the 'step' fields of both the
      timepoint_variable and the dependent_variable refer to a global step that
      can be used to align dependent_variable with timepoint_variable. If
      dependent_variable and timepoint_variable have different sets of step
      values, we only load those steps that exist in common for both variables.
      If False, we assume that the 'step' field of the dependent_variable refers
      to the actual values of the timepoint_variable, and the two variables are
      aligned on that instead. Whichever variable is used for alignment (global
      step or timepoints) are also used to determine where restarts occurred, so
      that any loose "tails" can be discarded if needed. This variable is
      irrelevant if timepoint_variable == None.

  Returns:
    A list of numpy arrays.

    If the input data are training curves, each numpy array represents a curve
    and has dims [2 x curve_length]. curve[0, :] are the timepoints and
    curve[1, :] are the dependent variable. Each curve is sorted by the
    timepoint variable.

    If the input data are sets of rollouts, each numpy array represents a single
    set of rollouts (e.g. n_rollouts for a single model checkpoint) and has dims
    [2 x n_rollouts]. rollouts[0, :] is just an index variable (e.g.
    range(0, n_rollouts)) and rollouts[1, :] are the performances per rollout.
    Each rollout set is sorted by the index variable.
  """

  if align_on_global_step:
    restart_determiner_x = 'step'
  else:
    restart_determiner_x = 'value'
  restart_determiner_y = 'step'

  curves = []
  for run_dir in run_dirs:
    accumulator = event_accumulator.EventAccumulator(
        run_dir, size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE)
    accumulator.Reload()

    # Load the dependent variable.
    y_vals, y_steps = extract_summary(accumulator, dependent_variable,
                                      restart_determiner_y)
    y_vals_dict = {step: val for step, val in zip(y_steps, y_vals)}

    # Load the timepoint variable.
    if timepoint_variable is None:
      # Load from the step values of y.
      steps_to_load = set(y_steps)
      x_steps = y_steps
      x_vals_dict = {step: step for step in x_steps}
    else:
      # Load from summaries.
      x_vals, x_steps = extract_summary(accumulator, timepoint_variable,
                                        restart_determiner_x)
      if align_on_global_step:
        x_vals_dict = {step: val for step, val in zip(x_steps, x_vals)}
      else:
        x_vals_dict = {val: val for val in x_vals}

      # Find steps in common between x and y
      if align_on_global_step:
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


def extract_summary(accumulator, summary_name, restart_determiner=None):
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
  values = [get_summary_value(summary, load_tensors) for summary in summaries]

  # Discard the "tails" from restarts.
  steps, values = discard_tails_from_restarts(steps, values, restart_determiner)

  return values, steps


def get_summary_value(summary, load_tensors):
  if load_tensors:
    proto = summary.tensor_proto
    value = tf.make_ndarray(proto)
    return value.item()
  else:
    return summary.value


def discard_tails_from_restarts(steps, values, determiner='step'):
  """Discard extra "tails" that are generated when there is a restart.

  Args:
    steps: List of global_steps loaded from summaries.
    values: List of values loaded from summaries.
    determiner: 'step' -- determine restarts based on the steps. 'value' --
      determine restarts based on the values. None -- do not discard any values.

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
