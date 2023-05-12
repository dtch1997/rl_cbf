# bump architecture with supervised losses
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id DiverseCartPole-v1 \
  --eval-frequency 1000 \
  --viz-frequency 1000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group debug \
  --save-model \
  --total-timesteps 15000 \
  --exploration-fraction 0.125 \
  --exp-name debug \
  --capture-video \
  --track \
  --seed 0 \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0