#!/bin/bash

for SEED in 1 2 3 4 5
do
    bash experiments/cartpole.sh $SEED
done