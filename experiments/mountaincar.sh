#!/bin/bash

# The variable $1 refers to the first argument passed to the script. 
# If no argument is passed, 'Default Value' will be used.
seed=${1:-'0'}
group_name=${2:-'mountaincar_ablations'}

echo "Running experiments with seed: $seed"

# baseline 
python rl_cbf/learning/dqn_mountaincar_train.py \
  --env-id BaseMountainCar-v0 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 1000000 \
  --exploration-fraction 0.125 \
  --capture-video \
  --track \
  --seed $seed \
  --exp-name baseline_1M \
  --max-value 0 \
  --min-value -100

# supervised
python rl_cbf/learning/dqn_mountaincar_train.py \
  --env-id BaseMountainCar-v0 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 1000000 \
  --exploration-fraction 0.125 \
  --capture-video \
  --track \
  --seed $seed \
  --exp-name supervised_1M \
  --max-value 0 \
  --min-value -100 \
  --supervised-loss-coef 1.0

# bump
python rl_cbf/learning/dqn_mountaincar_train.py \
  --env-id BaseMountainCar-v0 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 1000000 \
  --exploration-fraction 0.125 \
  --capture-video \
  --track \
  --seed $seed \
  --exp-name bump_1M \
  --max-value 0 \
  --min-value -100 \
  --enable-bump-parametrization

# siren
python rl_cbf/learning/dqn_mountaincar_train.py \
  --env-id BaseMountainCar-v0 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 1000000 \
  --exploration-fraction 0.125 \
  --capture-video \
  --track \
  --seed $seed \
  --exp-name siren_1M \
  --max-value 0 \
  --min-value -100 \
  --enable-siren-layer

# bump, supervised
python rl_cbf/learning/dqn_mountaincar_train.py \
  --env-id BaseMountainCar-v0 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 1000000 \
  --exploration-fraction 0.125 \
  --capture-video \
  --track \
  --seed $seed \
  --exp-name bump_supervised_1M \
  --max-value 0 \
  --min-value -100 \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0

# bump, siren
python rl_cbf/learning/dqn_mountaincar_train.py \
  --env-id BaseMountainCar-v0 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 1000000 \
  --exploration-fraction 0.125 \
  --capture-video \
  --track \
  --seed $seed \
  --exp-name bump_siren_1M \
  --max-value 0 \
  --min-value -100 \
  --enable-bump-parametrization \
  --enable-siren-layer

# siren, supervised
python rl_cbf/learning/dqn_mountaincar_train.py \
  --env-id BaseMountainCar-v0 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 1000000 \
  --exploration-fraction 0.125 \
  --capture-video \
  --track \
  --seed $seed \
  --exp-name siren_supervised_1M \
  --max-value 0 \
  --min-value -100 \
  --supervised-loss-coef 1.0 \
  --enable-siren-layer

# bump, siren, supervised
python rl_cbf/learning/dqn_mountaincar_train.py \
  --env-id BaseMountainCar-v0 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 1000000 \
  --exploration-fraction 0.125 \
  --capture-video \
  --track \
  --seed $seed \
  --exp-name bump_siren_supervised_1M \
  --max-value 0 \
  --min-value -100 \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0 \
  --enable-siren-layer