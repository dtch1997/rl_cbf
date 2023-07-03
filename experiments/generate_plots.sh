#!/bin/bash

bash experiments/visualization/viz_metrics.sh
bash experiments/visualization/viz_barrier.sh
bash experiments/evaluation/eval_constrain.sh

bash experiments/evaluation/eval_constrain_random.sh
bash experiments/evaluation/viz_constrain_random.sh

python experiments/visualization/viz_metric_tradeoff.py \
    -i experiments/artifacts/data/metrics.csv \
    -m1 eval/barrier/validity_alpha_0.9 \
    -m2 eval/barrier/coverage \
    -o experiments/artifacts/plots/viz_barrier_coverage_vs_validity.png