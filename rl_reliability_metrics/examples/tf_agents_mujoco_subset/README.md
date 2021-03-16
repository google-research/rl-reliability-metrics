# Example of running full pipeline: TF-Agents on OpenAI MuJoCo baselines subset

This example runs the full pipeline on a subset of the TF-Agents on OpenAI
MuJoCo baselines that were analyzed in the
[Measuring the Reliability of Reinforcement Learning Algorithms]
(https://arxiv.org/abs/1912.05663)
paper.

## Overview

### Data

We analyze a subset of the TF-Agents MuJoCo baselines from the paper.

* 3 algorithms (REINFORCE, SAC, TD3) and 2 tasks (Humanoid-v2 and Swimmer-v2)
* 3 training runs per (algo, task) combination

### Metrics evaluated

We only evaluate the "during training" metrics, i.e. risk and dispersion across
training runs and across time (within training runs).

## Steps

Parameters (paths, which metrics to evaluate, metric parameters, etc) have
already been defined in `params.py`.

NB: To analyze your own experiment results, you will need to customize
`./params.py` and the config file at
`rl_reliability_metrics/evaluation/example.gin`.

0. Download the example data.

   This package can ingest data in the form of CSV files or Tensorboard
   summaries. Set your desired data source at `data_source` in `params.py`.

   ```sh
   # This should match base_dir in params.py:
   BASE_DIR="$HOME/rl_reliability_metrics/tf_agents_mujoco_expts"

   mkdir -p $BASE_DIR/data
   cd $BASE_DIR/data
   ```

   To download the CSV example data:
   ```
   wget https://storage.googleapis.com/rl-reliability-metrics/data/tf_agents_example_csv_dataset.tgz
   tar -xvzf tf_agents_example_csv_dataset.tgz
   ```

   To download the Tensorboard example data:
   ```
   wget https://storage.googleapis.com/rl-reliability-metrics/data/tf_agents_example_dataset.tgz
   tar -xvzf tf_agents_example_dataset.tgz
   ```

0.  Navigate to the library root and set Python path:

    ```sh
    cd /your/installation/location/rl-reliability-metrics
    PYTHONPATH=.
    ```

0.  Evaluate the metrics on the training runs. This evaluates:

    (a) all the metrics on the original data

    (b) the across-run metrics on the data with training runs permuted for each
    pair of algorithms (this is necessary for the later step of performing
    permutation tests to compare algorithms)

    (c) the across-run metrics on the data with training runs resampled with
    replacement for each algorithm individually (this is necessary for the later
    step of computing bootstrap confidence intervals).

    For this example, we only run a small number of permutations and bootstrap
    resamples. In general we recommend >= 1000 permutations / bootstraps. If
    e.g. running on a cluster, you may use the flags `permutation_start_idx` and
    `bootstrap_start_idx` to assign a subset of permutations / bootstraps to
    each worker.

    The output of this step are three set of metric values (on the raw,
    permuted, and bootstrapped data).

    ```sh
    EXAMPLES=rl_reliability_metrics/examples
    python3 $EXAMPLES/tf_agents_mujoco_subset/evaluate_metrics.py
    python3 $EXAMPLES/tf_agents_mujoco_subset/evaluate_metrics.py --resampling permute
    python3 $EXAMPLES/tf_agents_mujoco_subset/evaluate_metrics.py --resampling bootstrap
    ```

0.  Compute permutation tests to determine whether algorithms are statistically
    different in their metric values. The output of this step is a p-value for
    each pair of algorithms.

    ```sh
    python3 $EXAMPLES/tf_agents_mujoco_subset/permutation_tests.py
    ```

0.  Compute bootstrap confidence intervals on the rankings of the algorithms.
    The output of this step is a confidence interval for each algorithm.

    ```sh
    python3 $EXAMPLES/tf_agents_mujoco_subset/bootstrap_confidence_intervals.py
    ```

0.  Make plots.

    One plots show the rankings of each algorithm (aggregated across tasks), for
    each metric. Horizontal bars above the algorithms indicate which pairs of
    algorithms are statistically different. Confidence intervals are indicated
    as well.

    Another set of plots show the per-task metric values for each algorithm, for
    each metric. Confidence intervals are indicated.

    ```sh
    python3 $EXAMPLES/tf_agents_mujoco_subset/plots.py
    ```
