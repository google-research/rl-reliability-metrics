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

"""Tests for data loading."""

import os
import tempfile

from absl import flags
from absl.testing import parameterized
import gin
import numpy as np
from rl_reliability_metrics.evaluation import data_loading
import tensorflow as tf

import unittest
from tensorboard.backend.event_processing import event_accumulator

FLAGS = flags.FLAGS


class DataLoadingTest(parameterized.TestCase, unittest.TestCase):

  def setUp(self):
    super(DataLoadingTest, self).setUp()

    gin.clear_config()
    gin_file = os.path.join(
        './',
        'rl_reliability_metrics/evaluation',
        'eval_metrics_test.gin')
    gin.parse_config_file(gin_file)

    # Initialize a DataLoader.
    self.data_loader = data_loading.DataLoader(
        dependent_variable='Metrics/AverageReturn',
        timepoint_variable='Metrics/EnvironmentSteps',
        discard_tails=True,
        align_on_global_step=True)

    # Fake sets of training curves to test analysis.
    test_data_dir = os.path.join(
        './',
        'rl_reliability_metrics/evaluation/test_data')
    self.tensorboard_dirs = [  # Tensorboard data.
        os.path.join(test_data_dir, 'tfsummary', 'run%d' % i, 'train')
        for i in range(3)
    ]
    self.csv_paths = [  # CSV data.
        os.path.join(test_data_dir, 'csv', 'run%d.csv' % i) for i in range(2)
    ]

  def test_load_csv(self):
    curves = self.data_loader.load_input_data(self.csv_paths)
    self.assertLen(curves, 2)
    self.assertEqual(curves[0].shape, (2, 3))
    # Check that tails are correctly discarded from each curve.
    expected_run0 = np.array([[1052, 2059, 3021], [183, 155, 201]])
    expected_run1 = np.array([[1020, 2028, 2038, 3031], [145, 129, 129, 126]])
    np.testing.assert_allclose(expected_run0, curves[0])
    np.testing.assert_allclose(expected_run1, curves[1])

  def test_load_tensorboard(self):
    curves = self.data_loader.load_input_data(self.tensorboard_dirs)
    self.assertLen(curves, 3)
    self.assertEqual(curves[0].shape, (2, 3))

  def test_load_curves_unordered(self):
    # Generate a curve that is unordered (according to env step).
    log_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
      for global_step, env_step, avg_return in [(0, 5, 5.1), (1, 3, 3.2),
                                                (2, 7, 7.3), (2, 9, 9.5)]:
        tf.summary.scalar(
            'Metrics/EnvironmentSteps', env_step, step=global_step)
        tf.summary.scalar('Metrics/AverageReturn', avg_return, step=global_step)
    # Test load_curves using steps of dependent variable as timepoint variable.
    # Check that, for repeated steps, only the last step is loaded.
    data_loader = data_loading.DataLoader(
        dependent_variable='Metrics/AverageReturn',
        timepoint_variable=None,
        discard_tails=True,
        align_on_global_step=True)
    curves = data_loader.load_input_data([log_dir])
    expected = np.array([[0, 1, 2], [5.1, 3.2, 9.5]])
    np.testing.assert_allclose(expected, curves[0])

    # Test load_curves using EnvironmentSteps as timepoint variable. Check that,
    # for repeated steps, only the last step is loaded, and that the curve is
    # now ordered.
    curves = self.data_loader.load_input_data([log_dir])
    expected = np.array([[3, 5, 9], [3.2, 5.1, 9.5]])
    np.testing.assert_allclose(expected, curves[0])

  def test_load_curves_steps_cleanup_on_global_step(self):
    # Generate a curve where the steps differ for the timepoint and dependent
    # variables, and where there are repeated values of the step. Use the global
    # step to align timepoint and dependent variables.
    log_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    writer = tf.summary.create_file_writer(log_dir)

    with writer.as_default():
      for global_step, env_step, avg_return in [(0, 3, 3.2), (1, 5, 5.1),
                                                (2, 5, 7.3), (2, 7, 7.4)]:
        tf.summary.scalar(
            'Metrics/EnvironmentSteps', env_step, step=global_step)
        tf.summary.scalar('Metrics/AverageReturn', avg_return, step=global_step)

      # Add an extra summary only for the timepoint variable, with no
      # corresponding summary for the dependent variable.
      tf.summary.scalar('Metrics/EnvironmentSteps', 10, step=3)

    # Test load_input_data, check that we only load the summaries that have step
    # values in common for both variables, and that we only load the latest
    # summary for each step value.
    curves = self.data_loader.load_input_data([log_dir])
    expected = np.array([[3, 5, 7], [3.2, 5.1, 7.4]])
    np.testing.assert_allclose(expected, curves[0])

  @parameterized.parameters(
      (False, np.array([[3, 7], [7.3, 7.4]])),
      (True, np.array([[10], [7.3]])),
  )
  def test_load_curves_steps_cleanup_on_timestep_variable(
      self, align_on_global_step, expected):
    # Generate a curve where the steps differ for the timepoint and dependent
    # variables, and where there are repeated values of the step. Use the
    # timepoint variable to align timepoint and dependent variables.
    log_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    with tf.summary.create_file_writer(log_dir).as_default():
      for global_step, env_step, avg_return in [(0, 5, 5.1), (1, 3, 3.2),
                                                (2, 3, 7.3), (2, 7, 7.4)]:
        # Write the timestep variable.
        tf.summary.scalar(
            'Metrics/EnvironmentSteps', env_step, step=global_step)
        # Write the dependent variable.
        tf.summary.scalar('Metrics/AverageReturn', avg_return, step=env_step)
        tf.summary.scalar('Metrics/EnvironmentSteps', 10, step=3)

    data_loader = data_loading.DataLoader(
        dependent_variable='Metrics/AverageReturn',
        timepoint_variable='Metrics/EnvironmentSteps',
        discard_tails=True,
        align_on_global_step=align_on_global_step)
    curves = data_loader.load_input_data([log_dir])
    np.testing.assert_allclose(expected, curves[0])

  def test_load_curves_with_restart_in_global_step(self):
    # Generate a curve where there is a restart in the global step variable.
    log_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    with tf.summary.create_file_writer(log_dir).as_default():
      # Write the timestep variable.
      for global_step, env_step in [(0, 10), (1, 20), (1, 21), (2, 30)]:
        tf.summary.scalar(
            'Metrics/EnvironmentSteps', env_step, step=global_step)

      # Write the dependent variable.
      for global_step, avg_return in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        tf.summary.scalar('Metrics/AverageReturn', avg_return, step=global_step)

    curves = self.data_loader.load_input_data([log_dir])
    expected = np.array([[10, 21, 30], [1, 2, 3]])
    np.testing.assert_allclose(expected, curves[0])

  def test_load_curves_with_restart_in_timepoints(self):
    # Generate a curve where there is a restart in the timepoint variable,
    # and where the dependent variable has two extra values.
    log_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    with tf.summary.create_file_writer(log_dir).as_default():
      # Write the timestep variable.
      for global_step, env_step in [(0, 10), (1, 21), (2, 20), (3, 30)]:
        tf.summary.scalar(
            'Metrics/EnvironmentSteps', env_step, step=global_step)

      # Write the dependent variable.
      for global_step, avg_return in [(10, 1), (20, 2), (21, 3), (20, 4),
                                      (30, 5), (40, 6)]:
        tf.summary.scalar('Metrics/AverageReturn', avg_return, step=global_step)

    data_loader = data_loading.DataLoader(
        dependent_variable='Metrics/AverageReturn',
        timepoint_variable='Metrics/EnvironmentSteps',
        discard_tails=True,
        align_on_global_step=False)
    curves = data_loader.load_input_data([log_dir])
    expected = np.array([[10, 20, 30], [1, 4, 5]])
    np.testing.assert_allclose(expected, curves[0])

  def test_extract_summary(self):
    accumulator = event_accumulator.EventAccumulator(self.tensorboard_dirs[0])
    accumulator.Reload()

    expected_summary_length = 3

    for summary_name in [
        'Metrics/AverageReturn', ['NotAValidKey', 'Metrics/AverageReturn']
    ]:
      values, steps = self.data_loader._extract_summary(accumulator,
                                                        summary_name)
      self.assertLen(values, expected_summary_length)
      self.assertLen(steps, expected_summary_length)

  def test_discard_tails_from_restarts_on_step(self):
    steps = [1, 2, 3, 4, 2, 3, 5, 6, 6, 6, 7, 9, 8, 8]
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    new_steps, new_values = self.data_loader._discard_tails_from_restarts(
        steps, values, determiner='step')
    np.testing.assert_array_equal(new_steps, [1, 2, 3, 5, 6, 7, 8])
    np.testing.assert_array_equal(new_values, [0, 4, 5, 6, 9, 10, 13])

  def test_discard_tails_from_restarts_on_value(self):
    steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    values = [1, 2, 3, 4, 2, 3, 5, 6, 6, 6, 7, 9, 8, 8]
    new_steps, new_values = self.data_loader._discard_tails_from_restarts(
        steps, values, determiner='value')
    np.testing.assert_array_equal(new_steps, [0, 4, 5, 6, 9, 10, 13])
    np.testing.assert_array_equal(new_values, [1, 2, 3, 5, 6, 7, 8])

  def test_discard_tails_from_restarts_no_restarts(self):
    steps = [1, 2, 3, 4, 2, 3, 5, 6, 6, 6, 7, 9, 8, 8]
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    new_steps, new_values = self.data_loader._discard_tails_from_restarts(
        steps, values, determiner='value')
    np.testing.assert_array_equal(new_steps, steps)
    np.testing.assert_array_equal(new_values, values)


if __name__ == '__main__':
  unittest.main()
