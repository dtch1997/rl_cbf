# CartPoleA
python rl_cbf/learning/dqn_decomposed_train.py \
  --env-id CartPoleA-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_supervised_2M \
  --capture-video \
  --track \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0

# CartPoleB
python rl_cbf/learning/dqn_decomposed_train.py \
  --env-id CartPoleB-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_supervised_2M \
  --capture-video \
  --track \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0