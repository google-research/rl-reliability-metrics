# Example of running full pipeline: OpenAI MuJoCo Baselines subset

This example runs the full pipeline on a subset of the Tf_agents MuJoCo
baselines that were analyzed in the
[Measuring the Reliability of Reinforcement Learning Algorithms](https://openreview.net/pdf?id=SJlpYJBKvH)
paper.

## Overview

### Data

We analyze a subset of the TF-Agents MuJoCo baselines from the paper.
* two algorithms and two tasks
* three training runs per (algo, task) combination

### Metrics evaluated

We only evaluate the "during training" metrics, i.e. risk and dispersion across
training runs and across time (within training runs).

## Steps

Parameters (paths, which metrics to evaluate, metric parameters, etc) have
already been defined in `./params.py`

TODO(scychan) add instructions for downloading the data

TODO(scychan) add instructions for navigating to the correct path

1.  Evaluate the metrics on the training runs. This evaluates (a) all the
    metrics on the original data (b) the across-run metrics on the data with
    training runs permuted for each pair of algorithms (this is necessary for
    the later step of performing permutation tests to compare algorithms) (c)
    the across-run metrics on the data with training runs resampled with
    replacement for each algorithm individually (this is necessary for the later
    step of computing bootstrap confidence intervals).

    For this example, we only run a small number of permutations and bootstrap
    resamples. In general we recommend >= 1000 permutations / bootstraps. If
    e.g. running on a cluster, you may use the flags `permutation_start_idx` and
    `bootstrap_start_idx` to assign a subset of permutations / bootstraps to
    each worker.

    The output of this step are three set of metric values (on the raw,
    permuted, and bootstrapped data).

    `python evaluate_metrics.py`

2.  Compute permutation tests to determine whether algorithms are statistically
    different in their metric values. The output of this step is a p-value for
    each pair of algorithms.

    `python permutation_tests.py`

3.  Compute bootstrap confidence intervals on the rankings of the algorithms.
    The output of this step is a confidence interval for each algorithm.

    `python bootstrap_confidence_intervals.py`

4.  Make plots.

    One plots show the rankings of each algorithm (aggregated across tasks), for
    each metric. Horizontal bars above the algorithms indicate which pairs of
    algorithms are statistically different. Confidence intervals are indicated
    as well.

    Another set of plots show the per-task metric values for each algorithm, for
    each metric. Confidence intervals are indicated.

    `python plots.py`
