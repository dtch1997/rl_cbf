#!/bin/bash

# The variable $1 refers to the first argument passed to the script. 
# If no argument is passed, 'Default Value' will be used.
env_type=${1:-"walker2d"}
group_name=${2:-'walker_ablate_options'}

env="Safety-${env_type}-random-v2"

for seed in 1 2 3 4 5
do 
    for bounded in True False 
    do 
        for unsafe_supervised in True False 
        do            
            python rl_cbf/offline/td3_bc_train.py \
                --env $env \
                --relabel zero_one \
                --project rl-cbf \
                --group $group_name \
                --seed $seed \
                --name $env-zero_one-seed=$seed \
                --max_timesteps 500000 \
                --checkpoints_path models \
                --buffer_size 3000000 \
                --use_mixed_dataset True \
                --bounded $bounded \
                --unsafe_supervised $supervised
        done
    done
done
