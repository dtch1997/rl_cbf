#!/bin/bash

# The variable $1 refers to the first argument passed to the script. 
# If no argument is passed, 'Default Value' will be used.
seed=${1:-'0'}
group_name=${2:-'d4rl_cbf'}

echo "Running experiments with seed: $seed"

for env in Safety-ant-medium-v2 Safety-walker2d-medium-v2 Safety-hopper-medium-v2
do 
    for relabel in identity zero_one
    do
        python rl_cbf/offline/td3_bc.py \
            --env $env \
            --relabel $relabel \
            --group $group_name \
            --seed $seed \
            --name $env-$relabel-seed=$seed
    done
done