
# Composite CBFs

Train checkpoints:
```
# 79mad5bw
ql_clbf/learning/dqn.py --env-id CartPoleC --track --wandb-entity dtch1997 --wandb-project QL-CLBF --save-model
# jmocxwi0
ql_clbf/learning/dqn.py --env-id CartPoleD --track --wandb-entity dtch1997 --wandb-project QL-CLBF --save-model
```

Videos available in videos folder

Reproduce results:

```
python ql_clbf/learning/dqn_eval.py --model-paths reports/composite_cbf/checkpoints/79mad5bw/dqn.pth reports/composite_cbf/checkpoints/jmocxwi0/dqn.pth
```