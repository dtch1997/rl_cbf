#!/bin/bash

# The variable $1 refers to the first argument passed to the script. 
# If no argument is passed, 'Default Value' will be used.
seed=${1:-'0'}
group_name=${2:-'cql'}

echo "Running experiments with seed: $seed"

for env_type in walker2d hopper ant 
do 
    for dataset in random medium medium-expert medium-replay
    do       
        env="Safety-${env_type}-${dataset}-v2"
        python rl_cbf/offline/cql_train.py \
            --env $env \
            --relabel zero_one \
            --project rl-cbf \
            --group $group_name \
            --seed $seed \
            --name $env-$relabel-seed=$seed \
    	    --max_timesteps 500000 \
            --checkpoints_path models
    done
done
