#!/bin/bash

# The variable $1 refers to the first argument passed to the script. 
# If no argument is passed, 'Default Value' will be used.
seed=${1:-'0'}
group_name=${2:-'d4rl_cbf'}

echo "Running experiments with seed: $seed"

for env in Safety-walker2d-medium-v2 # Safety-ant-medium-v2 Safety-hopper-medium-v2
do 
    for relabel in zero_one # identity constant_0.2 constant_0.8
    do
        for bounded in True False 
        do 
            for supervised in True False 
            do 
                python rl_cbf/offline/td3_bc.py \
                    --env $env \
                    --relabel $relabel \
                    --project rl-cbf \
                    --group $group_name \
                    --seed $seed \
                    --name $env-$relabel-seed=$seed \
                    --max_timesteps 150000 \
                    --checkpoints_path models \
                    --bounded $bounded \
                    --supervised $supervised
            done 
        done
    done
done
