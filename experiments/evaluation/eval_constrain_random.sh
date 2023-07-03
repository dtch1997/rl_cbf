#!/bin/bash
mkdir -p experiments/artifacts/data/eval_constrain_random
mkdir -p experiments/artifacts/plots

# Bump, supervised

experiment_names=(
    "bump_supervised_2M"
    "baseline_2M"
    "baseline_supervised_2M"
    "bump_2M"
    "bump_supervised_base_2M"
)

for exp_name in "${experiment_names[@]}"
do
    python experiments/evaluation/eval_constrain_random.py \
        --model-path experiments/artifacts/checkpoints/${exp_name}.pth \
        --enable-bump-parametrization \
        --device cpu \
        --save-path experiments/artifacts/data/eval_constrain_random/${exp_name} \
        --seed 1 \
        --num-rollouts 100
done