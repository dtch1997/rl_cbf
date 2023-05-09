""" DQN script to use with CartPole{A,B,C,D}-v1 """
import argparse
import os
import random
import time
from distutils.util import strtobool

import pandas as pd
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import rl_cbf.envs
from rl_cbf.learning.env_utils import make_env
from rl_cbf.learning.dqn_cartpole_eval import DQNCartPoleEvaluator
from rl_cbf.learning.dqn_cartpole_viz import DQNCartPoleVisualizer
from rl_cbf.net.q_network import QNetwork

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPoleA-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--eval-frequency", type=int, default=-1,
        help="the frequency of evaluation")
    parser.add_argument("--viz-frequency", type=int, default=-1,
        help="the frequency of visualization")
    
    # CLBF specific arguments
    parser.add_argument("--supervised-loss-coef", type=float, default=0.0)
    parser = QNetwork.add_argparse_args(parser)
    args = parser.parse_args()
    # fmt: on
    return args

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def save_model(model, model_path, base_path, args):
    torch.save(model.state_dict(), model_path)
    print(f"model saved to {model_path}")

    if args.track:
        wandb.save(model_path, base_path=base_path)
        print(f"model saved to wandb cloud")       

if __name__ == "__main__":
    args = parse_args()
    _env = gym.make(args.env_id)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork.from_argparse_args(envs, args).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork.from_argparse_args(envs, args).to(device)
    target_network.load_state_dict(q_network.state_dict())

    evaluator = DQNCartPoleEvaluator()
    visualizer = DQNCartPoleVisualizer()

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # Implement analytic loss
                # In CartPole, assume that state [0,0,0,0] is optimal
                safe_states = np.random.uniform(
                    low = (-2.0, 0, 0, 0),
                    high = (2.0, 0, 0, 0),
                    size = (args.batch_size, 4)
                ).astype(np.float32)
                safe_states = torch.Tensor(safe_states).to(device)
                optimal_qval_pred = q_network(safe_states)
                optimal_val_pred = optimal_qval_pred.max(dim=1)[0]
                optimal_val_true = 100 * torch.ones(args.batch_size, device=device, dtype=torch.float32)
                safe_loss = F.mse_loss(optimal_val_pred, optimal_val_true)
                writer.add_scalar("losses/safe_loss", safe_loss, global_step)
                loss += args.supervised_loss_coef * safe_loss

                # Implement unsafe loss
                states = np.random.uniform(
                    low = (-4.8, -4.0, -0.4, -3.0),
                    high = (4.8, 4.0, 0.4, 3.0),
                )
                # Use the base gym env to check if the state is unsafe
                is_unsafe = _env.is_done(states)
                unsafe_states = states[is_unsafe].astype(np.float32)
                unsafe_states = torch.from_numpy(unsafe_states).to(device).to(torch.float32)
                unsafe_qval_pred = q_network(unsafe_states)
                unsafe_val_pred = unsafe_qval_pred.max(dim=1)[0]
                unsafe_val_true = 0 * torch.ones(unsafe_val_pred.shape, device=device, dtype=torch.float32)
                unsafe_loss = F.mse_loss(unsafe_val_pred, unsafe_val_true)
                writer.add_scalar("losses/unsafe_loss", unsafe_loss, global_step)
                loss += args.supervised_loss_coef * unsafe_loss

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
            
            if args.eval_frequency > 0 and global_step % args.eval_frequency == 0:
                # TODO: evaluate model
                eval_strategies = ('rollout', 'grid')
                for strategy in eval_strategies:
                    metrics: pd.DataFrame = evaluator.evaluate(q_network, strategy)
                    writer.add_scalar(f'eval/{strategy}/episode_length', metrics['episode_length'].mean(), global_step)
                    writer.add_scalar(f'eval/{strategy}/episode_return', metrics['episode_return'].mean(), global_step)
                    writer.add_scalar(f'eval/{strategy}/values', metrics['mean_values'].mean(), global_step)
                    writer.add_scalar(f'eval/{strategy}/td_errors', metrics['mean_td_errors'].mean(), global_step)

            if args.viz_frequency > 0 and global_step % args.viz_frequency == 0: 
                fig = visualizer.visualize(q_network)
                writer.add_figure('viz/barrier_function', fig, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pth"
        base_path = f"runs/{run_name}"
        save_model(q_network, model_path, base_path, args) 
    
    envs.close()
    writer.close()