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

"""Utilities for plotting."""

import matplotlib.pyplot as plt
import numpy as np


def paper_figure_configs():
  plt.rcParams['axes.facecolor'] = 'white'
  plt.rcParams['font.family'] = 'serif'


def simple_axis(ax):
  # Hide the right and top spines
  ax.spines['left'].set_visible(True)
  ax.spines['bottom'].set_visible(True)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  # Only show ticks on the left and bottom spines
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')


def no_axis(ax):
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)


METRICS_DISPLAY_NAMES = {
    'HighFreqEnergyWithinRuns': 'High Frequency across Time (DT)',
    'IqrWithinRuns': 'IQR across Time (DT)',
    'MadWithinRuns': 'MAD across Time (DT)',
    'StddevWithinRuns': 'Stddev across Time (DT)',
    'LowerCVaROnDiffs': 'Lower CVaR on Differences (SRT)',
    'UpperCVaROnDiffs': 'Upper CVaR on Differences (SRT)',
    'MaxDrawdown': 'Max Drawdown (LRT)',
    'LowerCVaROnDrawdown': 'Lower CVaR on Drawdown (LRT)',
    'UpperCVaROnDrawdown': 'Upper CVaR on Drawdown (LRT)',
    'LowerCVaROnRaw': 'Lower CVaR on Raw',
    'UpperCVaROnRaw': 'Upper CVaR on Raw',
    'IqrAcrossRuns': 'IQR across Runs (DR)',
    'MadAcrossRuns': 'MAD across Runs (DR)',
    'StddevAcrossRuns': 'Stddev across Runs (DR)',
    'LowerCVaROnAcross': 'Lower CVaR across Runs (RR)',
    'UpperCVaROnAcross': 'Upper CVaR across Runs (RR)',
    'MedianPerfDuringTraining': 'Median Performance across Runs',
    'IqrAcrossRollouts': 'IQR across rollouts (DF)',
    'MadAcrossRollouts': 'MAD across rollouts (DF)',
    'LowerCVaRAcrossRollouts': 'Lower CVaR across rollouts (RF)',
    'UpperCVaRAcrossRollouts': 'Upper CVaR across rollouts (RF)',
    'MedianPerfAcrossRollouts': 'Median Performance across rollouts',
}

ALGO_ABBREVIATIONS = {
    'rainbow': 'RBW',
    'iqn': 'IQN',
    'c51': 'C51',
    'dqn': 'DQN',
    'ddpg': 'DDPG',
    'td3': 'TD3',
    'sac': 'SAC',
    'reinforce': 'REINF',
    'ppo': 'PPO',
}


def flipped_errorbar(x, y, yerr, ymax, bar_color, hatch_pattern, x_offset=0.6):
  """Plots a single bar for a y-axis whose values are descending from the top.

  Args:
    x: x-position of the bar.
    y: y-position of the bar.
    yerr: Tuple indicating (lower, upper) values on an error bar. These values
      are relative to zero, not relative to the y-position of the bar. Set to
      None for no errorbars.
    ymax: The largest value on the y-axis.
    bar_color: Color of the bar.
    hatch_pattern: Pattern to fill the bar.
    x_offset: Additional offset for the x-position.
  """
  plt.bar([x + x_offset], [ymax + 1 - y],
          color=bar_color,
          hatch=hatch_pattern,
          edgecolor='k')

  if yerr:
    yerr_flipped = np.array([[yerr[1] - ymax - 1],
                             [ymax + 1 - yerr[0]]])
    plt.errorbar([x + 1], [0], yerr_flipped, color='k')
