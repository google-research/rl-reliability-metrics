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

"""Base class for robustness metrics."""

import abc
import gin
import six


@gin.configurable
def all_metrics():
  # Only return the public classes.
  return Metric.public_subclasses()


@six.add_metaclass(abc.ABCMeta)
class Metric(object):
  """Abstract base class for robustness metrics."""

  @classmethod
  def public_subclasses(cls):
    """Recursively get all public subclasses."""
    all_subclasses = []
    for subclass in cls.__subclasses__():
      # Add this subclass only if it is public.
      if subclass.__name__[0] != '_':
        all_subclasses.append(subclass)
      # Recursively add the subclasses of the subclass.
      all_subclasses.extend(subclass.public_subclasses())
    return all_subclasses

  @property
  def name(self):
    """String name of the metric."""
    return type(self).__name__

  @abc.abstractproperty
  def result_dimensions(self):
    """String abbreviation describing the dimensions of the metric results.

    A-algorithm, T-task, R-run/rollout, P-evaluation point.
    """
    return NotImplementedError

  @abc.abstractproperty
  def bigger_is_better(self):
    """If True, more positive values are more desirable for the metric."""
    return NotImplementedError

  @abc.abstractmethod
  def __call__(self, curves):
    """Calls the metric for evaluation on a set of input data."""
    pass
