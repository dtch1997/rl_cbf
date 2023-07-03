#!/bin/bash
mkdir -p experiments/artifacts/data/eval_constrain
mkdir -p experiments/artifacts/data/eval_rollout
mkdir -p experiments/artifacts/plots

# Bump, supervised
python experiments/evaluation/eval_constrain.py \
    --model-path experiments/artifacts/checkpoints/bump_supervised_2M.pth \
    --enable-bump-parametrization \
    --device cpu \
    --plot-path experiments/artifacts/plots/eval_constrain_bump_supervised_2M.png \
    --save-path experiments/artifacts/data/eval_constrain/bump_supervised_2M.csv \
    --seed 1

python experiments/evaluation/eval_rollout.py \
    --model-path experiments/artifacts/checkpoints/bump_supervised_2M.pth \
    --enable-bump-parametrization \
    --device cpu \
    --save-path experiments/artifacts/data/eval_rollout/bump_supervised_2M.csv \
    --seed 1

python experiments/visualization/viz_constrain_x.py \
    --save-path experiments/artifacts/plots/eval_constrain_viz_x_constrained.png \
    --data-paths experiments/artifacts/data/eval_constrain/bump_supervised_2M.csv \
    experiments/artifacts/data/eval_rollout/bump_supervised_2M.csv \
    --labels "Constrained" "Rollout"