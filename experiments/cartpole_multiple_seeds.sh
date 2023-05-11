#!/bin/bash

group_name=${1:-'cartpole_ablations'}

for SEED in 1 2 3 4 5
do
    bash experiments/cartpole.sh $SEED $group_name
done