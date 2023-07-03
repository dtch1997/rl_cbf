#!/bin/bash

python experiments/visualization/viz_barrier.py \
    --model-paths \
        experiments/artifacts/checkpoints/bump_supervised_2M.pth \
        experiments/artifacts/checkpoints/baseline_2M.pth \
        experiments/artifacts/checkpoints/baseline_supervised_2M.pth \
        experiments/artifacts/checkpoints/bump_2M.pth \
        experiments/artifacts/checkpoints/bump_supervised_base_2M.pth \
    --device cpu \
    --save-path experiments/artifacts/plots/viz_barrier.png \
