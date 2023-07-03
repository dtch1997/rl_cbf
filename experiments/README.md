# Experiments

Collection of scripts to reproduce experiments presented in the paper. 

Run all scripts from project root directory (containing `License`, `setup.py` etc), i.e. `/path/to/rl_cbf`

## Training

Run experiments on CartPole:

```bash
bash experiments/training/cartpole_multiple_seeds.sh 
```

## Evaluation

Here we explain how to reproduce figures from the paper. 

One pre-trained checkpoint per experimental setting is available in `experiments/artifacts/checkpoints`. 

Metrics for 5 seeded runs of each experimental setting is available in `experiments/artifacts/data/metrics.csv`

Collection of pre-generated plots is available in `experiments/artifacts/plots`. They are generated as follows:

```bash
bash experiments/generate_plots.sh
``` 