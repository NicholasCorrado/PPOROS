# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import copy
import os
import pickle
import random
import time
from collections import defaultdict
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
# from torch.utils.tensorboard import SummaryWriter

from evaluate import Evaluate
from utils import NormalizeObservation, NormalizeReward, get_latest_run_id


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v4", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="total timesteps of the experiments")
    parser.add_argument("--beta", type=int, default=False, help="Sample actions from Beta distribution rather than Gaussian")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4, help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--buffer-history", "-b", type=int, default=2, help="Number of prior collect phases to store in buffer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    parser.add_argument("--ros", type=float, default=True, help="True = use ROS policy to collect data, False = use target policy")
    parser.add_argument("--ros-learning-rate", "-ros-lr", type=float, default=1e-4, help="the learning rate of the ROS optimizer")
    parser.add_argument("--ros-clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--ros-num-minibatches", type=int, default=32, help="the number of mini-batches")
    parser.add_argument("--ros-reset-freq", type=int, default=1, help="Reset ROS policy to target policy every ros_reset_freq updates")
    parser.add_argument("--ros-update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument("--ros-mixture-prob", type=float, default=1, help="Probability of sampling ROS policy")
    parser.add_argument("--ros-update-freq", type=int, default=1)
    parser.add_argument("--ros-ent-coef", type=float, default=0.0, help="coefficient of the entropy in ros update")
    parser.add_argument("--ros-target-kl", type=float, default=0.01, help="the target KL divergence threshold")
    parser.add_argument("--ros-max-kl", type=float, default=None, help="the target KL divergence threshold")
    parser.add_argument("--ros-num-actions", type=int, default=10, help="the target KL divergence threshold")
    parser.add_argument("--ros-lambda", type=float, default=0.01, help="the target KL divergence threshold")
    parser.add_argument("--ros-uniform-sampling", type=bool, default=True, help="the target KL divergence threshold")
    parser.add_argument("--compute-sampling-error", type=int, default=False, help="True = use ROS policy to collect data, False = use target policy")

    parser.add_argument("--eval-freq", type=int, default=10, help="evaluate target and ros policy every eval_freq updates")
    parser.add_argument("--eval-episodes", type=int, default=20, help="number of episodes over which policies are evaluated")
    parser.add_argument("--results-dir", "-f", type=str, default="results", help="directory in which results will be saved")
    parser.add_argument("--results-subdir", "-s", type=str, default="", help="results will be saved to <results_dir>/<env_id>/<subdir>/")
    parser.add_argument("--policy-path", type=str, default=None, help="Path to pretrained policy")
    parser.add_argument("--normalization-dir", type=str, default=None, help="Directory contatining normalization stats")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.ros_minibatch_size = int(args.batch_size // args.ros_num_minibatches)

    # fmt: on

    if args.seed is None:
        args.seed = np.random.randint(2 ** 32 - 1)

    if args.ros:
        save_dir = f"{args.results_dir}/{args.env_id}/ppo_ros/{args.results_subdir}"
    else:
        save_dir = f"{args.results_dir}/{args.env_id}/ppo_buffer/{args.results_subdir}"

    if args.run_id:
        save_dir += f"/run_{args.run_id}"
    else:
        run_id = get_latest_run_id(save_dir=save_dir) + 1
        save_dir += f"/run_{run_id}"
    args.save_dir = save_dir

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    return args


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, action_mean, action_std, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x),

    def get_action(self, x, noise=False):
        action_mean = self.actor_mean(x)
        if noise:
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
        else:
            action = action_mean
        return action

    def get_action_and_info(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, action_mean, action_std, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def sample_actions(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        return action

    def sample_actions_unif(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Uniform(low=torch.clamp(action_mean-3*action_std,-1,+1), high=torch.clamp(action_mean+3*action_std,-1,+1))
        action = probs.sample()
        return action

class AgentBeta(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.net = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2*np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.action_dim = envs.single_action_space.shape[0]

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        net_output = self.net(x)
        alpha = 2 + torch.exp(net_output[:, :self.action_dim])
        beta = 2 + torch.exp(net_output[:, self.action_dim:])
        # print(torch.mean(torch.norm(alpha, dim=-1)), torch.mean(torch.norm(beta, dim=-1)))
        probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()
        return action, alpha, beta, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x):
        net_output = self.net(x)
        alpha = 2 + torch.exp(net_output[:, :self.action_dim])
        beta = 2 + torch.exp(net_output[:, self.action_dim:])
        action_mean = alpha / (alpha+beta)
        return action_mean

def update_ppo(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer):
    # flatten the batch
    b_obs = obs[:global_step].reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs[:global_step].reshape(-1)
    b_actions = actions[:global_step].reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages[:global_step].reshape(-1)
    b_returns = returns[:global_step].reshape(-1)
    b_values = values[:global_step].reshape(-1)

    # Optimizing the policy and value network
    batch_size = b_obs.shape[0]
    minibatch_size = int(batch_size // args.num_minibatches)

    b_inds = np.arange(batch_size)
    clipfracs = []
    skipped_updates = 0
    for epoch in range(args.update_epochs):
        approx_kls = []
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                approx_kls.append(approx_kl)

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        avg_kl = np.mean(approx_kls)

        if args.track:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("ppo/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("ppo/value_loss", v_loss.item(), global_step)
            writer.add_scalar("ppo/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("ppo/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("ppo/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("ppo/approx_kl", avg_kl, global_step)
            writer.add_scalar("ppo/epochs", epoch+1, global_step)
            writer.add_scalar("ppo/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("ppo/explained_variance", explained_var, global_step)
            # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # print('ppo:', np.mean(approx_kls))

        if args.target_kl is not None:
            if avg_kl > args.target_kl:
                break

    ppo_stats = {
        't': global_step,
        'approx_kl': avg_kl,
        'clip_frac': np.mean(clipfracs),
        'policy_loss': pg_loss.item(),
        'entropy': entropy_loss.item(),
    }

    return ppo_stats


def update_ros(agent_ros, agent, envs, ros_optimizer, obs, logprobs, actions, global_step, args, buffer_size, writer):

    # flatten the batch
    if global_step < buffer_size:
        end = global_step
    else:
        end = -args.num_steps

    # flatten the batch
    b_obs = obs[:end].reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs[:end].reshape(-1)
    b_actions = actions[:end].reshape((-1,) + envs.single_action_space.shape)

    # threshold_prob = 1e-5
    # b_probs = b_logprobs.exp()
    # mask = b_probs > threshold_prob
    # b_obs = b_obs[mask]
    # b_logprobs = b_logprobs[mask]
    # b_actions = b_actions[mask]

    # Optimizing the policy and value network
    batch_size = b_obs.shape[0]
    minibatch_size = int(batch_size // args.ros_num_minibatches)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    skipped_updates = 0

    # b_probs = b_probs[mask]
    # probs_min = b_probs.min()
    # probs_max = b_probs.max()
    # probs_count = mask.sum()
    # probs_mean = b_probs.mean()
    #
    # if args.track:
    #     writer.add_scalar("ppo/probs_min", probs_min, global_step)
    #     writer.add_scalar("ppo/probs_max", probs_max, global_step)
    #     writer.add_scalar("ppo/probs_count", probs_count, global_step)
    #     writer.add_scalar("ppo/probs_mean", probs_mean, global_step)

    for epoch in range(args.ros_update_epochs):
        approx_kls = []
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, _, _, ros_logprob, entropy = agent_ros.get_action_and_info(b_obs[mb_inds], b_actions[mb_inds])
            ros_logratio = ros_logprob - b_logprobs[mb_inds]
            ros_ratio = ros_logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                # old_approx_kl = (-logratio).mean()
                approx_kl = ((ros_ratio - 1) - ros_logratio).mean()
                approx_kls.append(approx_kl.item())
                clipfracs += [((ros_ratio - 1.0).abs() > args.ros_clip_coef).float().mean().item()]
                # print(approx_kl)

                # if approx_kl > args.ros_target_kl:
                #     print(global_step, epoch, approx_kl.item())

            # loss = None
            # if args.ros_max_kl is not None:
            #     if approx_kl > args.ros_max_kl:
            #         skipped_updates += 1
            #         continue

            pushup_loss = 0
            if args.ros_num_actions:
                for i in range(args.ros_num_actions):
                    if args.ros_uniform_sampling:
                        random_actions = torch.rand_like(b_actions[mb_inds])*2 - 1 # random actions in [-1, +1]
                    else:
                        with torch.no_grad():
                            random_actions = agent.sample_actions(b_obs[mb_inds])
                    _, _, _, pushup_logprob, entropy = agent_ros.get_action_and_info(b_obs[mb_inds], random_actions)
                    # push_up_loss = torch.clamp(newlogprob, 1 - args.ros_clip_coef, 1 + args.ros_clip_coef)
                    # push_up_loss += push_up_loss.mean()
                    # pushup_logratio = pushup_logprob - ros_logprob
                    # pushup_ratio = pushup_logratio.exp()
                    # push_up_loss += pushup_ratio.mean()

                    # pushup_prob = pushup_logprob.exp()
                    # pushup_loss1 = pushup_prob
                    # pushup_loss2 = torch.clamp(pushup_ratio, 1 - args.ros_clip_coef, 0.03)
                    # pushup_loss += torch.max(pushup_loss1, pushup_loss2).mean()
                    # pushup_loss += torch.clamp(pushup_prob, 1e-3, 0.03).mean()
                    # pushup_loss += pushup_prob.mean()

                    pushup_loss += pushup_logprob.mean()

                pushup_loss = pushup_loss/args.ros_num_actions

            # Policy loss
            pg_loss1 = ros_ratio
            pg_loss2 = torch.clamp(ros_ratio, 1 - args.ros_clip_coef, 1 + args.ros_clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            # pg_loss = pg_loss2.mean()
            # pg_loss = torch.clamp(pg_loss, 0.9, 1.1)

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ros_lambda*pushup_loss - args.ros_ent_coef * entropy_loss

            ros_optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(agent_ros.parameters(), args.max_grad_norm)
            ros_optimizer.step()

        avg_kl = np.mean(approx_kls)
        if args.track:
            writer.add_scalar("ros/learning_rate", ros_optimizer.param_groups[0]["lr"], global_step)
            # writer.add_scalar("ros/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("ros/approx_kl", avg_kl, global_step)
            writer.add_scalar("ros/epochs", epoch+1, global_step)
            writer.add_scalar("ros/skipped_updates", skipped_updates, global_step)
            writer.add_scalar("ros/clipfrac", np.mean(clipfracs), global_step)

            if loss:
                writer.add_scalar("ros/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("ros/entropy", entropy_loss.item(), global_step)
            if pushup_loss:
                writer.add_scalar("ros/pushup_loss", pushup_loss.item(), global_step)
        # print(approx_kl)
        # print('ros:', np.mean(approx_kls))
        # print(approx_kls)
        if args.ros_target_kl:
            if avg_kl > args.ros_target_kl:
                print('Early stop:', avg_kl, args.ros_target_kl)
                # print(approx_kls)
                break

    ros_stats = {
        't': global_step,
        'approx_kl': avg_kl,
        'clip_frac': np.mean(clipfracs),
        'policy_loss': pg_loss.item(),
        'entropy': entropy_loss.item(),
    }
    if pushup_loss:
        ros_stats['pushup_loss'] = pushup_loss.item()

    return ros_stats


def normalize_obs(obs_rms, obs):
    """Normalises the observation using the running mean and variance of the observations."""
    return torch.Tensor((obs.detach().numpy() - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8))

def normalize_reward(return_rms, rewards):
    """Normalizes the rewards with the running mean rewards and their variance."""
    return rewards / np.sqrt(return_rms.var + 1e-8)

def main():
    args = parse_args()
    run_name = (args.save_dir).replace('/','_')
    if args.track:
        import wandb
        from torch.utils.tensorboard import SummaryWriter

        wandb.login(key="7313077863c8908c24cc6058b99c2b2cc35d326b")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True, no longer works for gymnasium
            save_code=True,
        )
        writer = SummaryWriter(f"wandb/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # PPO target agent
    if args.beta:
        agent = AgentBeta(envs).to(device)
    else:
        agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.policy_path:
        agent = torch.load(args.policy_path)

    # ROS behavior agent
    agent_ros = copy.deepcopy(agent)  # initialize ros policy to be equal to the eval policy
    ros_optimizer = optim.Adam(agent_ros.parameters(), lr=args.ros_learning_rate, eps=1e-5)

    # Evaluation modules
    eval_module = Evaluate(model=agent, eval_env=None, n_eval_episodes=args.eval_episodes, log_path=args.save_dir, device=device)
    eval_module_ros = Evaluate(model=agent_ros, eval_env=None, n_eval_episodes=args.eval_episodes, log_path=args.save_dir, device=device, suffix='ros')


    history_k = args.buffer_history
    buffer_size = history_k * args.num_steps
    buffer_pos = 0
    # ALGO Logic: Storage setup
    # ALGO Logic: Storage setup
    obs_buffer = torch.zeros((buffer_size, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buffer = torch.zeros((buffer_size, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards_buffer = torch.zeros((buffer_size, args.num_envs)).to(device)
    dones_buffer = torch.zeros((buffer_size, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    timesteps = []
    sampling_error = []
    kl_ros_target = []
    entropy_target = []
    entropy_ros = []

    ppo_logs = {
        't': [],
        'approx_kl': [],
        'clip_frac': [],
        'policy_loss': [],
        'entropy': [],
    }
    ros_logs = {
        't': [],
        'approx_kl': [],
        'clip_frac': [],
        'policy_loss': [],
        'entropy': [],
        'pushup_loss': [],
    }
    ppo_logs = defaultdict(lambda: [])
    ros_logs = defaultdict(lambda: [])

    # agent_mle = copy.deepcopy(agent)
    # optimizer_mle = optim.Adam(agent_mle.parameters(), lr=1e-3, eps=1e-5)

    env_reward_normalize = envs.envs[0].env
    env_obs_normalize = envs.envs[0].env.env.env

    if args.normalization_dir:
        with open(f'{args.normalization_dir}/env_obs_normalize', 'rb') as f:
            obs_rms = pickle.load(f)
            env_obs_normalize.obs_rms = obs_rms

        with open(f'{args.normalization_dir}/env_reward_normalize', 'rb') as f:
            return_rms = pickle.load(f)
            env_reward_normalize.return_rms = return_rms

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

            # args.ros_clip_coef = frac * args.ros_clip_coef
            # ros_lrnow = frac * args.ros_learning_rate
            # ros_optimizer.param_groups[0]["lr"] = ros_lrnow

        loss = 0
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs_buffer[buffer_pos] = env_obs_normalize.unnormalize(next_obs) # store unnormalized obs
            dones_buffer[buffer_pos] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.ros and np.random.random() < args.ros_mixture_prob:
                    action, action_mean, action_std, logprob_ros, entropy, _ = agent_ros.get_action_and_value(next_obs)
                    if args.track:
                        writer.add_scalar("ros/action_mean", action_mean.detach().mean().item(), global_step)
                        writer.add_scalar("ros/action_std", action_std.detach().mean().item(), global_step)
                else:
                    action, action_mean, action_std, _, _, _ = agent.get_action_and_value(next_obs)
                actions_buffer[buffer_pos] = action


            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            reward = env_reward_normalize.unnormalize(reward)
            rewards_buffer[buffer_pos] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            buffer_pos += 1
            buffer_pos %= buffer_size

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue

                if args.track:
                    writer.add_scalar("charts/ros_train_ret", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episode_length", info["episode"]["l"], global_step)

        if global_step < buffer_size:
            indices = np.arange(buffer_pos)
        else:
            indices = (np.arange(buffer_size) + buffer_pos) % buffer_size

        # obs = normalize_obs(obs_rms, obs_buffer.clone()[indices])
        obs = obs_buffer[indices]
        actions = actions_buffer[indices]
        rewards = rewards_buffer[indices]
        dones = dones_buffer[indices]

        env_obs_normalize.set_update(False)
        obs = env_obs_normalize.normalize(obs).float()
        env_obs_normalize.set_update(True)

        env_obs_normalize.set_update(False)
        rewards = env_reward_normalize.normalize(rewards).float()
        env_obs_normalize.set_update(True)

        # assert torch.isnan(obs).sum() == 0
        # assert torch.isnan(rewards).sum() == 0

        with torch.no_grad():
            _, _, _, new_logprob, _, new_value = agent.get_action_and_value(obs.reshape(-1, obs_dim), actions.reshape(-1, action_dim))
            values = new_value
            logprobs = new_logprob.reshape(-1, 1)

        # bootstrap value if not done
        # Compute returns and advantages -- bootstrap value if not done
        with torch.no_grad():
            # next_value = agent.get_value(normalize_obs(obs_rms, next_obs)).reshape(1, -1)
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            num_steps = indices.shape[0]
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        if update % args.update_freq == 0:
            ppo_stats = update_ppo(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer)
            for key, val in ppo_stats.items():
                ppo_logs[key].append(ppo_stats[key])
        if args.ros and (update % args.ros_update_freq == 0):# and global_step > 25000:
            # Set ROS policy equal to current target policy
            if update % args.ros_reset_freq == 0:
                for source_param, dump_param in zip(agent_ros.parameters(), agent.parameters()):
                    source_param.data.copy_(dump_param.data)

            # ROS behavior update
            ros_stats = update_ros(agent_ros, agent, envs, ros_optimizer, obs, logprobs, actions, global_step, args, buffer_size, writer)
            for key, val in ros_stats.items():
                ros_logs[key].append(ros_stats[key])
        # if update % 2 == 0:
        #     update_ppo(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer)
        if update % args.eval_freq == 0:
            current_time = time.time() - start_time
            print(f"Training time: {int(current_time)} \tsteps per sec: {int(global_step / current_time)}")
            target_ret, target_std = eval_module.evaluate(global_step, train_env=envs, noise=False)
            ros_ret, ros_std = eval_module_ros.evaluate(global_step, train_env=envs, noise=False)
            #
            # print('With noise:')
            # target_ret, target_std = eval_module.evaluate(global_step, train_env=envs, noise=True)
            # ros_ret, ros_std = eval_module_ros.evaluate(global_step, train_env=envs, noise=True)

            np.savez(f'{args.save_dir}/ppo_stats.npz', **ppo_logs)
            np.savez(f'{args.save_dir}/ros_stats.npz', **ros_logs)


            if args.track:
                writer.add_scalar("charts/ppo_eval_return", target_ret, global_step)
                writer.add_scalar("charts/ros_eval_return", ros_ret, global_step)

            if args.compute_sampling_error:
                agent_mle = copy.deepcopy(agent)
                optimizer_mle = optim.Adam(agent_mle.parameters(), lr=1e-3)

                b_obs = obs.reshape(-1, obs_dim)
                b_actions = actions.reshape(-1, action_dim)

                for i in range(10000):
                    _, logprobs_mle, _ = agent_mle.get_action_and_info(b_obs, b_actions)
                    loss = -torch.mean(logprobs_mle)

                    # if i % 100 == 0:
                    #     print(i, loss)

                    optimizer_mle.zero_grad()
                    loss.backward()
                    optimizer_mle.step()

                with torch.no_grad():
                    _, logprobs_target, ent_target = agent.get_action_and_info(b_obs, b_actions)
                    logratio = logprobs_mle - logprobs_target
                    ratio = logratio.exp()
                    approx_kl_mle_target = ((ratio - 1) - logratio).mean()
                    print('D_kl( mle || target ) = ', approx_kl_mle_target.item())
                    _, logprobs_ros, ent_ros = agent_ros.get_action_and_info(b_obs, b_actions)

                    logratio = logprobs_ros - logprobs_target
                    ratio = logratio.exp()
                    approx_kl_ros_target = ((ratio - 1) - logratio).mean()
                    print('D_kl( ros || target ) = ', approx_kl_ros_target.item())

                    sampling_error.append(approx_kl_mle_target.item())
                    kl_ros_target.append(approx_kl_ros_target.item())
                    entropy_target.append(ent_target.mean().item())
                    entropy_ros.append(ent_ros.mean().item())
                    timesteps.append(global_step)

                    np.savez(f'{args.save_dir}/stats.npz',
                             t=timesteps,
                             sampling_error=sampling_error,
                             kl_ros_target=kl_ros_target,
                             entropy_target=entropy_target,
                             entropy_ros=entropy_ros)
    envs.close()

if __name__ == "__main__":
    main()