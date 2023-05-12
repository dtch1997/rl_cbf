#!/bin/bash

group_name=${1:-'mountaincar_ablations'}

for SEED in 1 2 3 4 5
do
    bash experiments/mountaincar.sh $SEED $group_name
done