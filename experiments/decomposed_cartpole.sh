# CartPoleA
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id CartPoleA-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group cartpole_decompose_ab \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_supervised_2M \
  --capture-video \
  --track \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0

# CartPoleB
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id CartPoleB-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group cartpole_decompose_ab \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_supervised_2M \
  --capture-video \
  --track \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0

  # CartPoleC
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id CartPoleC-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group cartpole_decompose_cd \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_supervised_2M \
  --capture-video \
  --track \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0

  # CartPoleD
python rl_cbf/learning/dqn_cartpole_train.py \
  --env-id CartPoleD-v1 \
  --eval-frequency 10000 \
  --viz-frequency 100000 \
  --wandb-entity dtch1997 \
  --wandb-project RL_CBF \
  --wandb-group cartpole_decompose_cd \
  --save-model \
  --total-timesteps 2000000 \
  --exploration-fraction 0.125 \
  --exp-name bump_supervised_2M \
  --capture-video \
  --track \
  --enable-bump-parametrization \
  --supervised-loss-coef 1.0