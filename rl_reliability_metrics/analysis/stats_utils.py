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

"""Utitilies for statistics."""

from absl import logging
import scipy.special


def multiple_comparisons_correction(n_algorithms,
                                    alpha,
                                    method='benjamini-yekutieli'):
  """Compute corrected p-value thresholds to account for multiple comparisons.

  Types of correction methods:
  - Benjamini-Yekutieli correction:
    https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Yekutieli_procedure
  - Bonferroni correction:
    https://en.wikipedia.org/wiki/Bonferroni_correction

  Args:
    n_algorithms: Number of algorithms.
    alpha: Uncorrected p-value threshold.
    method: 'benjamini-yekutieli' or 'bonferroni'

  Returns:
    If method == 'benjamini-yekutieli':
      A list of float values with length (number of possible pairs of
      algorithms), i.e. length (number of possible comparisons).
      The i'th float is the corrected p-value threshold for the pair of
      algorithms with the i'th largest uncorrected p-value.
    If method == 'bonferroni':
      A float value for the corrected p-value threshold.
  """
  n_algo_pairs = int(scipy.special.comb(n_algorithms, 2))

  if method == 'benjamini-yekutieli':
    c = 1
    pthresh_corrected = [1 / n_algo_pairs * alpha]
    for i in range(2, n_algo_pairs + 1):
      c += 1 / i
      pthresh_corrected.append(i / (n_algo_pairs * c) * alpha)

  elif method == 'bonferroni':
    pthresh_corrected = alpha / n_algo_pairs

  else:
    raise ValueError('Invalid multiple comparisons correction method: %s' %
                     method)

  logging.info('Corrected p-value thresholds: %s', pthresh_corrected)
  return pthresh_corrected
