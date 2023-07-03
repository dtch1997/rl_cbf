#!/bin/bash

# Training history, rollout episode return
python experiments/visualization/viz_metric_train_hist.py \
    -i experiments/artifacts/data/metrics.csv \
    -m eval/rollout/episode_return \
    -o experiments/artifacts/plots/viz_rollout_episode_return.png

# Training history, grid mean td error
python experiments/visualization/viz_metric_train_hist.py \
    -i experiments/artifacts/data/metrics.csv \
    -m eval/grid/mean_td_error \
    -o experiments/artifacts/plots/viz_grid_td_error.png

# Training history, barrier coverage
python experiments/visualization/viz_metric_train_hist.py \
    -i experiments/artifacts/data/metrics.csv \
    -m eval/barrier/coverage \
    -o experiments/artifacts/plots/viz_barrier_coverage.png

# Training history, barrier validity
python experiments/visualization/viz_metric_train_hist.py \
    -i experiments/artifacts/data/metrics.csv \
    -m eval/barrier/validity_alpha_0.9 \
    -o experiments/artifacts/plots/viz_barrier_validity.png

# Training history, constrain length
python experiments/visualization/viz_metric_train_hist.py \
    -i experiments/artifacts/data/metrics.csv \
    -m eval/constrain/mean_episode_length \
    -o experiments/artifacts/plots/viz_constrain_episode_length.png

