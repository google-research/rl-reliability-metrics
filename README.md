# RL Reliability Metrics

The RL Reliability Metrics library provides a set of metrics for measuring the
reliability of reinforcement learning (RL) algorithms. The library also provides
statistical tools for computing confidence intervals and for comparing
algorithms on these metrics.

As input, this library accepts a set of RL training curves, or a set of rollouts
of an already trained RL algorithm. The library computes reliability metrics
across different dimensions (additionally, it can also analyze non-reliability
metrics like median performance), and outputs plots presenting the reliability
metrics for each algorithm, aggregated across tasks or on a per-task basis. The
library also provides statistical tests for comparing algorithms based on these
metrics, and provides bootstrapped confidence intervals of the metric values.

## Table of contents

<a href='#Paper'>Paper</a><br>
<a href='#Installation'>Installation</a><br>
<a href='#Examples'>Examples</a><br>
<a href='#Datasets'>Datasets</a><br>
<a href='#Contributing'>Contributing</a><br>
<a href='#Principles'>Principles</a><br>
<a href='#Disclaimer'>Disclaimer</a><br>

## Paper

Please see the paper for a detailed description of the metrics and statistical
tools implemented by the RL Reliability Metrics library, and for examples of
applying the methods to common tasks and algorithms:
[Measuring the Reliability of Reinforcement Learning Algorithms.](https://arxiv.org/abs/1912.05663)

If you use this code or reference the paper, please cite it as:

```
@conference{rl_reliability_metrics,
  title = {Measuring the Reliability of Reinforcement Learning Algorithms},
  author = {Stephanie CY Chan, Sam Fishman, John Canny, Anoop Korattikara, and Sergio Guadarrama},
  booktitle = {International Conference on Learning Representations, Addis Ababa, Ethiopia},
  year = 2020,
}
```

## Installation

```bash
git clone https://github.com/google-research/rl-reliability-metrics
cd rl-reliability-metrics
pip3 install -r requirements.txt
```

Note: Only Python 3.x is supported.

## Examples

See
[`rl_reliability_metrics/examples/tf_agents_mujoco_subset`](rl_reliability_metrics/examples/tf_agents_mujoco_subset)
for an example of applying the full pipeline to a small example dataset.

## Datasets

For the continuous control dataset that was analyzed in the
[Measuring the Reliability of Reinforcement Learning Algorithms](https://arxiv.org/abs/1912.05663)
paper (TF-Agents algorithm implementations evaluated on OpenAI MuJoCo
baselines), please download using
[this URL](https://storage.googleapis.com/rl-reliability-metrics/data/tf_agents_full_dataset.tgz).

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for a guide on how to contribute.

## Principles

This project adheres to [Google's AI principles](PRINCIPLES.md). By
participating, using or contributing to this project you are expected to adhere
to these principles.

## Acknowledgements

Many thanks to Toby Boyd for his assistance in the open-sourcing process, Oscar
Ramirez for code reviews, and Pablo Castro for his help with running experiments
using the Dopamine baselines data. Thanks also to the following people for
helpful discussions during the formulation of these metrics and the writing of
the paper: Mohammad Ghavamzadeh, Yinlam Chow, Danijar Hafner, Rohan Anil, Archit
Sharma, Vikas Sindhwani, Krzysztof Choromanski, Joelle Pineau, Hal Varian,
Shyue-Ming Loh, and Tim Hesterberg.

## Disclaimer

This is not an official Google product.
