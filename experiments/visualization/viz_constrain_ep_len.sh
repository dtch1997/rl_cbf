#!/bin/bash

python experiments/visualization/viz_constrain_ep_len.py \
    --data-paths \
        experiments/artifacts/data/eval_constrain_random/bump_supervised_base_2M_ep_lengths.csv \
        experiments/artifacts/data/eval_constrain_random/bump_supervised_2M_ep_lengths.csv \
        experiments/artifacts/data/eval_constrain_random/bump_2M_ep_lengths.csv \
        experiments/artifacts/data/eval_constrain_random/baseline_supervised_2M_ep_lengths.csv \
        experiments/artifacts/data/eval_constrain_random/baseline_2M_ep_lengths.csv \
    --labels \
        "NOEXP" \
        "SIGMOID_SUP" \
        "SIGMOID" \
        "MLP_SUP" \
        "MLP" \
    --save-path experiments/artifacts/plots/viz_eval_constrain_ep_len \