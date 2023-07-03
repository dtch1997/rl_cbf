#!/bin/bash

# The variable $1 refers to the first argument passed to the script. 
# If no argument is passed, 'Default Value' will be used.
seed=${1:-'0'}
group_name=${2:-'cartpole_ablations'}

echo "Running experiments with seed: $seed"

# baseline experiment
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id DiverseCartPole-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name baseline_2M \
  --capture-video \
  --track \
  --seed $seed

# supervised losses
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id DiverseCartPole-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name baseline_supervised_2M \
  --capture-video \
  --track \
  --seed $seed \
  --supervised-loss-coef 1.0
  
# bump architecture
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id DiverseCartPole-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_2M \
  --capture-video \
  --track \
  --seed $seed \
  --enable-bump-parametrization

# bump architecture with supervised losses
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id DiverseCartPole-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_supervised_2M \
  --capture-video \
  --track \
  --seed $seed \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0

# ablate exploration by training on regular environment
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id BaseCartPole-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group $group_name \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_supervised_base_2M \
  --capture-video \
  --track \
  --seed $seed \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0

