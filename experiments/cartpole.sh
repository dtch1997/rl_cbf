# baseline experiment
python rl_cbf/learning/dqn_train.py \
  --env-id DiverseCartPole-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group DiverseCartPole \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name baseline_2M \
  --capture-video \
  --track

# supervised losses
python rl_cbf/learning/dqn_train.py \
  --env-id DiverseCartPole-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group DiverseCartPole \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name baseline_supervised_2M \
  --capture-video \
  --track \
  --supervised-loss-coef 1.0
  
# bump architecture
python rl_cbf/learning/dqn_train.py \
  --env-id DiverseCartPole-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group DiverseCartPole \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_2M \
  --capture-video \
  --track \
  --enable-bump-parametrization

# bump architecture with supervised losses
python rl_cbf/learning/dqn_train.py \
  --env-id DiverseCartPole-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group DiverseCartPole \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_supervised_2M \
  --capture-video \
  --track \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0

