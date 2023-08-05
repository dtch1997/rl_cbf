#!/bin/bash

# The variable $1 refers to the first argument passed to the script. 
# If no argument is passed, 'Default Value' will be used.
seed=${1:-'0'}
group_name=${2:-'d4rl_cbf'}

for seed in 1 2 3 4 5
do 
    for env in Safety-walker2d-medium-v2 # Safety-ant-medium-v2 Safety-hopper-medium-v2
    do 
        for relabel in zero_one # constant_0.2 constant_0.8
        do
            if [ $relabel == "zero_one" ]
            then
                options="--bounded True --supervised True --detach_actor True"
            else
                options="--bounded False --supervised False --detach_actor False"
            fi
            
            python rl_cbf/offline/td3_bc_train.py \
                --env $env \
                --relabel $relabel \
                --project rl-cbf \
                --group $group_name \
                --seed $seed \
                --name $env-$relabel-seed=$seed \
                --max_timesteps 500000 \
                --checkpoints_path models \
                $options
        done
    done
done