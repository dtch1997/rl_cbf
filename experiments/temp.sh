group_name=${1:-'cartpole_ablations_seeded'}

for seed in 1 2 3 4 5
do
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
    --exp-name bump_supervised_2M \
    --capture-video \
    --track \
    --seed $seed \
    --enable-bump-parametrization \
    --supervised-loss-coef 1.0
done