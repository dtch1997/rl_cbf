# Supervised Losses

We add a supervised loss on the unsafe states 

The supervised loss has the effect of enforcing the barrier condition

## Figures

We add a supervised loss on the unsafe states 

The supervised loss has the effect of enforcing the barrier condition

We plot:
1. Value trajectory (It should stay high around 100)
2. Barrier function values (The safe states should be accurate)

Compare with and without the supervised loss:

Without:

![](figures/plot_barrier_baseline.png)
![](figures/plot_value_baseline.png)

With:

![](figures/plot_barrier_unsafe.png)
![](figures/plot_value_unsafe.png)

## Diverse starting locations

Attempted to increase certified safe region by increasing the diversity of starting locations

Plots:
1. Value history
2. TD error history
3. Barrier function

Without anything else:

![](figures/plot_value_diverse.png)
![](figures/plot_td_error_diverse.png)
![](figures/plot_barrier_diverse.png)


With supervised loss on unsafe states as well as optimal states:

![](figures/plot_value_diverse_analytic_unsafe.png)
![](figures/plot_td_error_diverse_analytic_unsafe.png)
![](figures/plot_barrier_diverse_analytic_unsafe.png)

We observe that adding the supervised loss comes at the cost of increasing the TD error. 
Maybe I can achieve the same effect by simply adding a large negative reward on termination

## Videos

See respective `checkpoints/xxxx/rollout.mp4` 

Checkpoint trained with supervised loss: `vgdae80s`
![](checkpoints/vgdae8os/rollout.mp4)

Checkpoint trained without supervised loss: `yp8twvo`
![](checkpoints/yp8tywvo/rollout.mp4)

The checkpoint trained with supervised loss is clearly more stable

Checkpoint trained on DiverseCartPole without supervised losses: `yudtt505`
Checkpoint trained on DiverseCartPole with supervised losses: `fknvr3v5`