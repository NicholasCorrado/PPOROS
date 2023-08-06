import argparse
import copy
import os
import pickle
import random
import time
from collections import defaultdict, deque
from distutils.util import strtobool

import gymnasium as gym
import custom_envs
# import minatar
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from utils import Evaluate, AgentDiscrete, EvaluateDiscrete
from utils import get_latest_run_id, make_env, Agent

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        return env

    return thunk


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    # We use integers 0/1 instead of booleans False/True simply because the server we use for all experiments may
    # interpret False/True as strings instead of booleans.

    # weights and biases (wandb) parameters. Wandb is disabled by default.
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="If toggled, this experiment will be tracked with Weights and Biases (wandb)")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="Wandb experiment name")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb project entity (team)")
    parser.add_argument("--wandb-login-key", type=str, default=None, help="Wandb login key")

    # Saving and logging parameters
    parser.add_argument("--log-stats", type=int, default=1, help="If true, training statistics are logged")
    parser.add_argument("--eval-freq", type=int, default=1,
                        help="Evaluate PPO and/or PROPS policy every eval_freq PPO updates")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of episodes over which policies are evaluated")
    parser.add_argument("--results-dir", "-f", type=str, default="results",
                        help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")
    parser.add_argument("--results-subdir", "-s", type=str, default="",
                        help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")

    # General training parameters (both PROPS and PPO)
    parser.add_argument("--env-id", type=str, default="Bandit-v0", help="Environment id")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Number of timesteps to train")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="If toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="If toggled, cuda will be enabled by default")

    # PPO hyperparameters
    parser.add_argument("--num-steps", type=int, default=64,
                        help="PPO target batch size (n in paper), the number of steps to collect between each PPO policy update")
    parser.add_argument("--buffer-batches", "-b", type=int, default=1,
                        help="Number of PPO target batches to store in the replay buffer (b in paper)")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4, help="PPO Adam optimizer learning rate")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for PPO policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="General advantage estimation lambda (not the lambda used for PROPS")
    parser.add_argument("--num-minibatches", type=int, default=32, help="Number of minibatches updates for PPO update")
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of epochs for PPO update")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization for PPO update")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="Surrogate clipping coefficient \epsilon for PPO update")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy loss coefficient for PPO update")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient for PPO update")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum norm for gradient clipping for PPO update")
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="Target/cutoff KL divergence threshold for PPO update")

    # PROPS/ROS hyperparameters
    parser.add_argument("--props", type=int, default=1,
                        help="If True, use PROPS to collect data, otherwise use on-policy sampling")
    parser.add_argument("--ros", type=int, default=0,
                        help="If True, use ROS to collect data, otherwise use on-policy sampling")
    parser.add_argument("--props-num-steps", type=int, default=32,
                        help="PROPS behavior batch size (m in paper), the number of steps to run in each environment per policy rollout")
    parser.add_argument("--props-learning-rate", "-props-lr", type=float, default=1e-3,
                        help="PROPS Adam optimizer learning rate")
    parser.add_argument("--props-anneal-lr", type=lambda x: bool(strtobool(x)), default=0, nargs="?", const=False,
                        help="Toggle learning rate annealing for PROPS policy")
    parser.add_argument("--props-clip-coef", type=float, default=0.3,
                        help="Surrogate clipping coefficient \epsilon_PROPS for PROPS")
    parser.add_argument("--props-max-grad-norm", type=float, default=0.5,
                        help="Maximum norm for gradient clipping for PROPS update")
    parser.add_argument("--props-num-minibatches", type=int, default=4,
                        help="Number of minibatches updates for PROPS update")
    parser.add_argument("--props-update-epochs", type=int, default=4, help="Number of epochs for PROPS update")
    parser.add_argument("--props-target-kl", type=float, default=0.1,
                        help="Target/cutoff KL divergence threshold for PROPS update")
    parser.add_argument("--props-lambda", type=float, default=0.3, help="Regularization coefficient for PROPS update")
    parser.add_argument("--props-adv", type=int, default=False, help="If True, the PROPS update is weighted using the absolute advantage |A(s,a)|")
    parser.add_argument("--props-eval", type=int, default=False,
                        help="If set, the PROPS policy is evaluated every props_eval ")

    # Sampling error (se)
    parser.add_argument("--se", type=int, default=0,
                        help="If True, sampling error is computed every se_freq PPO updates.")
    parser.add_argument("--se-ref", type=int, default=0,
                        help="If True, on-policy sampling error is computed using the PPO policy sequence obtained while using PROPS. Only applies if se is True.")
    parser.add_argument("--se-lr", type=float, default=1e-3,
                        help="Adam optimizer learning rate used to compute the empirical (maximum likelihood) policy in sampling error computation.")
    parser.add_argument("--se-epochs", type=int, default=250,
                        help="Number of epochs to compute empirical (maximum likelihood) policy.")
    parser.add_argument("--se-freq", type=int, default=None, help="Compute sampling error very se_freq PPO updates")

    # loading pretrained models
    parser.add_argument("--policy-path", type=str, default=None, help="Path of pretrained policy to load")
    parser.add_argument("--normalization-dir", type=str, default=None,
                        help="Directory contatining normalization statistics of pretrained policy")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.buffer_size = args.buffer_batches * args.batch_size
    args.minibatch_size = int(args.buffer_size // args.num_minibatches)
    args.props_minibatch_size = int((args.buffer_size - args.props_num_steps) // args.props_num_minibatches)

    # cuda support. Currently does not work with normalization
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.seed is None:
        if args.run_id:
            args.seed = np.random.randint(args.run_id)
        else:
            args.seed = np.random.randint(2 ** 32 - 1)

    if args.props:
        save_dir = f"{args.results_dir}/{args.env_id}/ppo_props/{args.results_subdir}"
        if args.ros:
            save_dir = f"{args.results_dir}/{args.env_id}/ros/{args.results_subdir}"
    else:
        save_dir = f"{args.results_dir}/{args.env_id}/ppo_buffer/{args.results_subdir}"

    if args.run_id is not None:
        save_dir += f"/run_{args.run_id}"
    else:
        run_id = get_latest_run_id(save_dir=save_dir) + 1
        save_dir += f"/run_{run_id}"
    args.save_dir = save_dir

    # dump training config to save dir
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    return args


def update_ppo(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer):
    # PPO UPDATE

    # flatten buffer data
    b_obs = obs[:global_step].reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs[:global_step].reshape(-1)
    b_actions = actions[:global_step].reshape((-1,) + envs.single_action_space.shape).long()
    b_advantages = advantages[:global_step].reshape(-1)
    b_returns = returns[:global_step].reshape(-1)
    b_values = values[:global_step].reshape(-1)

    batch_size = b_obs.shape[0]  # number of transitions in replay buffer
    minibatch_size = min(args.minibatch_size, batch_size)

    b_inds = np.arange(batch_size)
    clipfracs = []
    grad_norms = []
    done_updating = False
    num_update_minibatches = 0
    approx_kl_to_log = None

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Stop updating if we exceed KL threshold.
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        done_updating = True
                        break

                approx_kl_to_log = approx_kl

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

            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            grad_norms.append(grad_norm.detach().cpu().numpy())
            optimizer.step()

            num_update_minibatches += 1

        if done_updating:
            break

    ppo_stats = {}

    # get update statistics if at least one minibatch update was performed.
    if num_update_minibatches > 0:
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        ppo_stats = {
            # 't': global_step,
            'ppo_value_loss': float(v_loss.item()),
            'ppo_policy_loss': float(pg_loss.item()),
            'ppo_entropy': float(entropy_loss.item()),
            'ppo_old_approx_kl': float(old_approx_kl.item()),
            'ppo_epochs': epoch + 1,
            'ppo_num_update_minibatches': float(num_update_minibatches),
            'ppo_clip_frac': float(np.mean(clipfracs)),
            'ppo_explained_variance': float(explained_var),
            'ppo_grad_norm': float(np.mean(grad_norms))
        }
        if approx_kl_to_log:
            ppo_stats['ppo_approx_kl'] = float(approx_kl_to_log.item())

        if args.track:
            writer.add_scalar("ppo/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("ppo/value_loss", v_loss.item(), global_step)
            writer.add_scalar("ppo/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("ppo/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("ppo/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("ppo/approx_kl", approx_kl_to_log, global_step)
            writer.add_scalar("ppo/epochs", epoch + 1, global_step)
            writer.add_scalar("ppo/num_update_minibatches", num_update_minibatches, global_step)
            writer.add_scalar("ppo/clip_frac", np.mean(clipfracs), global_step)
            writer.add_scalar("ppo/explained_variance", explained_var, global_step)
            writer.add_scalar("ppo/grad_norm", np.mean(grad_norms), global_step)

    return ppo_stats


def update_props(agent_props, envs, props_optimizer, obs, logprobs, actions, advantages, global_step, args, writer, logits):
    # PROPS UPDATE

    if global_step <= args.buffer_size - args.props_num_steps:
        # If the replay buffer is not full, use all data in replay buffer for this update.
        start = 0
        end = global_step
    else:
        # If the replay buffer is full, exclude the oldest behavior batch from this update; that batch will be evicted
        # before the next update and thus does not contribute to sampling error.
        start = args.props_num_steps
        end = args.buffer_size

    # flatten the replay buffer data
    b_obs = obs[start:end].reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs[start:end].reshape(-1)
    b_actions = actions[start:end].reshape((-1,) + envs.single_action_space.shape)
    # b_logits = logits[start:end].reshape(-1)  # action logits for PPO policy
    b_probs = np.exp(logprobs)

    if args.props_adv:
        b_advantages = advantages[start:end].reshape(-1)

    batch_size = b_obs.shape[0]
    minibatch_size = min(args.props_minibatch_size, batch_size)
    b_inds = np.arange(batch_size)
    clipfracs = []

    done_updating = False
    num_update_minibatches = 0
    pg_loss = None
    kl_regularizer_loss = None
    approx_kl_to_log = None
    grad_norms = []

    for epoch in range(args.props_update_epochs):
        np.random.shuffle(b_inds)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions[mb_inds]
            mb_probs = b_probs[mb_inds]
            mb_logprobs = b_logprobs[mb_inds]

            if args.props_adv:
                # Do not zero-center advantages; we need to preserve A(s,a) = 0 for AW-PROPS
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - 0) / (mb_advantages.std() + 1e-8)
                mb_abs_advantages = torch.abs(mb_advantages)
                # print(torch.mean(mb_abs_advantages), torch.std(mb_abs_advantages))

            _, _, props_logprobs, entropy = agent_props.get_action_and_info(mb_obs, mb_actions)
            props_logratio = props_logprobs - b_logprobs[mb_inds]
            props_ratio = props_logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-props_logratio).mean()
                approx_kl = ((props_ratio - 1) - props_logratio).mean()
                clipfracs += [((props_ratio - 1.0).abs() > args.props_clip_coef).float().mean().item()]

                if args.props_target_kl:
                    if approx_kl > args.props_target_kl:
                        done_updating = True
                        break
                approx_kl_to_log = approx_kl

            kl_regularizer_loss = (mb_probs*(mb_logprobs - props_logprobs)).mean()

            pg_loss1 = props_ratio
            pg_loss2 = torch.clamp(props_ratio, 1 - args.props_clip_coef, 1 + args.props_clip_coef)
            if args.props_adv:
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_abs_advantages).mean()
            else:
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if args.ros:
                pg_loss = -props_logratio.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss + args.props_lambda * kl_regularizer_loss

            props_optimizer.zero_grad()
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(agent_props.parameters(), args.props_max_grad_norm)
            grad_norms.append(grad_norm.detach().cpu().numpy())

            props_optimizer.step()
            num_update_minibatches += 1

        if done_updating:
            break

    props_stats = {}

    # Return training statistics if at least one minibatch update was performed.
    if num_update_minibatches > 0:
        props_stats = {
            'props_policy_loss': float(pg_loss.item()),
            'props_entropy': float(entropy_loss.item()),
            'props_old_approx_kl': float(old_approx_kl.item()),
            'props_epochs': epoch + 1,
            'props_clip_frac': float(np.mean(clipfracs)),
            'props_grad_norm': float(np.mean(grad_norms)),
            'props_num_update_minibatches': num_update_minibatches,
            # 'props_kl_regularizer_loss': float(kl_regularizer_loss.item()),
            'props_approx_kl': float(approx_kl_to_log.item()),
        }
        if args.track:
            writer.add_scalar("props/learning_rate", props_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("props/epochs", epoch + 1, global_step)
            writer.add_scalar("props/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("props/num_update_minibatches", num_update_minibatches, global_step)
            writer.add_scalar("props/grad_norm", np.mean(grad_norms), global_step)
            writer.add_scalar("props/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("props/entropy", entropy_loss.item(), global_step)
            # writer.add_scalar("props/kl_regularizer_loss", kl_regularizer_loss.item(), global_step)
            writer.add_scalar("props/approx_kl", approx_kl_to_log, global_step)
    return props_stats


def compute_se(args, agent, agent_props, obs, actions, advantages, sampling_error_logs, global_step, envs, prefix=""):
    # COMPUTE SAMPLING ERROR

    # Initialize empirical policy equal to the current PPO policy.
    agent_mle = copy.deepcopy(agent)

    # Freeze the feature layers of the empirical policy (as done in the Robust On-policy Sampling (ROS) paper)
    params = [p for p in agent_mle.actor_mean.parameters()]
    params[0].requires_grad = False
    params[2].requires_grad = False

    optimizer_mle = optim.Adam(agent_mle.parameters(), lr=args.se_lr)

    obs_dim = obs.shape[-1]
    action_dim = actions.shape[-1]
    b_obs = obs.reshape(-1, obs_dim).to(args.device)
    b_actions = actions.reshape(-1, action_dim).to(args.device)
    b_advantages = advantages.reshape(-1).to(args.device)

    n = len(b_obs)
    b_inds = np.arange(n)

    mb_size = 512 * b_obs.shape[0] // args.num_steps
    for epoch in range(args.se_epochs):

        np.random.shuffle(b_inds)
        for start in range(0, n, mb_size):
            end = start + mb_size
            mb_inds = b_inds[start:end]

            _, _, logprobs_mle, _ = agent_mle.get_action_and_info(b_obs[mb_inds], b_actions[mb_inds], clamp=True)
            loss = -torch.mean(logprobs_mle)

            optimizer_mle.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent_mle.parameters(), 1, norm_type=2)
            optimizer_mle.step()

    with torch.no_grad():
        _, action_probs, logprobs_mle, _ = agent_mle.get_action_and_info(b_obs, b_actions, clamp=True)

        # Compute sampling error
        _, action_probs_target, mean_target, std_target, logprobs_target, ent_target = agent.get_action_and_info(b_obs, b_actions, clamp=True)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        b_advantages = torch.abs(b_advantages)
        logratio = logprobs_mle - logprobs_target
        approx_kl_mle_target = (b_advantages * logratio).mean()

        # Compute KL divergence between PROPS and PPO policy
        _, action_probs_props, logprobs_props, ent_props = agent_props.get_action_and_info(b_obs, b_actions, clamp=True)
        logratio = logprobs_target - logprobs_props
        approx_kl_props_target = (torch.abs(b_advantages) * logratio).mean()

        sampling_error_logs[f'{prefix}kl_mle_target'].append(approx_kl_mle_target.item())
        sampling_error_logs[f'{prefix}kl_props_target'].append(approx_kl_props_target.item())
        sampling_error_logs[f'{prefix}ent_target'].append(ent_target.mean().item())
        sampling_error_logs[f'{prefix}ent_props'].append(ent_props.mean().item())

        np.savez(f'{args.save_dir}/stats.npz',
                 **sampling_error_logs)


def compute_se_ref(args, agent_buffer, envs, next_obs_buffer, sampling_error_logs, global_step):
    # COMPUTE ON-POLICY SAMPLING ERROR USING THE TARGET POLICY SEQUENCE OBTAINED DURING TRAINING

    envs_se = copy.deepcopy(envs)

    obs_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs_se.single_observation_space.shape).to(args.device)
    actions_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs_se.single_action_space.shape).to(args.device)
    buffer_pos = 0  # index of buffer position to be updated in the current timestep

    env_reward_normalize = envs_se.envs[0].env
    env_obs_normalize = envs_se.envs[0].env.env.env

    env_obs_normalize.set_update(False)
    env_reward_normalize.set_update(False)

    for i in range(len(agent_buffer)):
        agent = agent_buffer[i]
        if i == 0:
            next_obs = next_obs_buffer[i]

        for t in range(args.num_steps):
            obs_buffer[buffer_pos] = next_obs  # store normalized obs

            with torch.no_grad():
                action, action_probs, _, _, _ = agent.get_action_and_value(next_obs)
                actions_buffer[buffer_pos] = action

            next_obs, reward, terminated, truncated, infos = envs_se.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(args.device)

            buffer_pos += 1

    compute_se(args, agent_buffer[-1], agent_buffer[-1], obs_buffer[:buffer_pos], actions_buffer[:buffer_pos],
               sampling_error_logs, global_step, envs, prefix="ref_")


def main():
    args = parse_args()
    run_name = (args.save_dir).replace('/', '_')
    if args.track:
        import wandb
        from torch.utils.tensorboard import SummaryWriter

        # wandb.login(key=args.wandb_login_key)
        wandb.login(key='7313077863c8908c24cc6058b99c2b2cc35d326b')
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

    # seeding
    print('seed:', args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    capture_video = False
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only continuous action space is supported"
    env_reward_normalize = envs.envs[0].env
    env_obs_normalize = envs.envs[0].env.env.env

    # PPO target agent
    agent = AgentDiscrete(envs).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # load pretrained policy and normalization information
    if args.policy_path:
        agent = torch.load(args.policy_path).to(args.device)
    if args.normalization_dir:
        with open(f'{args.normalization_dir}/env_obs_normalize', 'rb') as f:
            obs_rms = pickle.load(f)
            env_obs_normalize.obs_rms = obs_rms
        with open(f'{args.normalization_dir}/env_reward_normalize', 'rb') as f:
            return_rms = pickle.load(f)
            env_reward_normalize.return_rms = return_rms

    # ROS behavior agent
    agent_props = copy.deepcopy(agent).to(args.device)  # initialize props policy to be equal to the eval policy
    props_optimizer = optim.Adam(agent_props.parameters(), lr=args.props_learning_rate, eps=1e-5)

    # Evaluation modules
    eval_module = EvaluateDiscrete(model=agent, eval_env=None, n_eval_episodes=args.eval_episodes, log_path=args.save_dir,
                           device=args.device)
    eval_module_props = EvaluateDiscrete(model=agent_props, eval_env=None, n_eval_episodes=args.eval_episodes,
                                 log_path=args.save_dir, device=args.device, suffix='props')

    # replay buffer setup
    obs_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    actions_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_action_space.shape).to(args.device)
    rewards_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(args.device)
    dones_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(args.device)
    buffer_pos = 0  # index of buffer position to be updated in the current timestep

    # initialize RL loop
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    num_updates = args.total_timesteps // args.batch_size
    if args.se_freq is None:
        args.se_freq = num_updates // 30
    # There are (ppo num updates)/(props num updates) times as many props updates.
    num_props_updates = num_updates * (args.num_steps // args.props_num_steps)

    agent_buffer = deque(maxlen=args.buffer_batches)
    next_obs_buffer = deque(maxlen=args.buffer_batches)

    # logging
    ppo_logs = defaultdict(lambda: [])
    props_logs = defaultdict(lambda: [])
    sampling_error_logs = defaultdict(lambda: [])

    global_step = 0
    start_time = time.time()
    target_update = 0

    # evaluate initial policy
    eval_module.evaluate(global_step, train_env=envs, noise=False)
    if args.props_eval:
        eval_module_props.evaluate(global_step, train_env=envs, noise=False)

    ppo_stats = {}
    props_stats = {}

    for props_update in range(1, num_props_updates + 1):
        for step in range(0, args.props_num_steps):
            global_step += 1 * args.num_envs
            obs_buffer[buffer_pos] = next_obs  # store unnormalized obs
            dones_buffer[buffer_pos] = next_done

            with torch.no_grad():
                if args.props:
                    action, action_probs, logprob_props, entropy, _ = agent_props.get_action_and_value(next_obs)
                    # if args.track:
                    #     writer.add_scalar("props/action_mean", action_mean.detach().mean().item(), global_step)
                    #     writer.add_scalar("props/action_std", action_std.detach().mean().item(), global_step)
                else:
                    action, action_probs, _, _, _ = agent.get_action_and_value(next_obs)
                actions_buffer[buffer_pos] = action

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards_buffer[buffer_pos] = torch.tensor(reward).to(args.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(done).to(args.device)

            buffer_pos += 1
            buffer_pos %= args.buffer_size

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue

                # if args.track:
                #     writer.add_scalar("charts/props_train_ret", info["episode"]["r"], global_step)
                #     writer.add_scalar("charts/episode_length", info["episode"]["l"], global_step)

        # Reorder the replay buffer from youngest to oldest so we can reuse cleanRL's code to compute advantages
        if global_step < args.buffer_size:
            indices = np.arange(buffer_pos)
        else:
            indices = (np.arange(args.buffer_size) + buffer_pos) % args.buffer_size
        obs = obs_buffer[indices]
        actions = actions_buffer[indices]
        rewards = rewards_buffer[indices]
        dones = dones_buffer[indices]

        # Recompute logprobs of all (s,a) in the replay buffer before every update, since they change after every update.
        with torch.no_grad():
            _, logits, new_logprob, _, new_value = agent.get_action_and_value(obs, actions)
            values = new_value.reshape(-1, envs.num_envs)
            logprobs = new_logprob.reshape(-1, envs.num_envs)

        # Store the b previous target policies. We do this so we can compute on-policy sampling error with respect to
        # the target policy sequence obtained by PROPS.
        if global_step % args.num_steps == 0:
            next_obs_buffer.append(copy.deepcopy(next_obs))
            agent_buffer.append(copy.deepcopy(agent))

        # Compute advantages and returns. If props_adv = True, then we must recompute advantages before every PROPS update, not just every target update.
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(args.device)
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

        # Compute sampling error *before* updating the target policy.
        if args.se:
            if global_step % (args.num_steps * args.se_freq) == 0:
                compute_se(args, agent, agent_props, obs, actions, advantages, sampling_error_logs, global_step, envs)
                if args.se_ref:
                    compute_se_ref(args, agent_buffer, envs, next_obs_buffer, sampling_error_logs, global_step)
                    sampling_error_logs[f'diff_kl_mle_target'].append(
                        sampling_error_logs[f'kl_mle_target'][-1] - sampling_error_logs[f'ref_kl_mle_target'][-1])
                    print('(PROPS - On-policy) sampling error:', sampling_error_logs[f'diff_kl_mle_target'])
                    print('On-policy sampling error:', sampling_error_logs[f'ref_kl_mle_target'])

                sampling_error_logs['t'].append(global_step)
                print('PROPS sampling error:', sampling_error_logs[f'kl_mle_target'])
                np.savez(f'{args.save_dir}/stats.npz',
                         **sampling_error_logs)

                if args.track:
                    writer.add_scalar("charts/diff_kl_mle_target", sampling_error_logs[f'diff_kl_mle_target'][-1],
                                      global_step)

        if args.track:
            best_arm_count = (actions_buffer.detach().numpy() == 999).sum()
            writer.add_scalar("charts/best_arm_count", best_arm_count, global_step)

        # PPO update
        if global_step % args.num_steps == 0 and args.update_epochs > 0:

            target_update += 1

            # Annealing learning rate
            if args.anneal_lr:
                frac = 1.0 - (target_update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # perform PPO update
            ppo_stats = update_ppo(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args,
                                   global_step, writer)

        # PROPS update
        if args.props and global_step > args.buffer_size and global_step % args.props_num_steps == 0:  # and global_step > 25000:
            # Annealing learning rate
            if args.props_anneal_lr:
                frac = 1.0 - (props_update - 1.0) / num_props_updates
                lrnow = frac * args.props_learning_rate
                props_optimizer.param_groups[0]["lr"] = lrnow

            # Set props policy equal to current target policy
            for source_param, dump_param in zip(agent_props.parameters(), agent.parameters()):
                source_param.data.copy_(dump_param.data)

            # perform props behavior update and log stats
            props_stats = update_props(agent_props, envs, props_optimizer, obs, logprobs, actions, advantages, global_step, args,
                                       writer, logits)

        # Evaluate agent performance
        if global_step % (args.num_steps * args.eval_freq) == 0:
            current_time = time.time() - start_time
            print(f"Training time: {int(current_time)} \tsteps per sec: {int(global_step / current_time)}")
            agent = agent.to(args.device)
            agent_props = agent_props.to(args.device)
            # Evaluate PPO policy
            target_ret, target_std = eval_module.evaluate(global_step, train_env=envs, noise=False)
            if args.props_eval:
                # Evaluate PROPS policy
                props_ret, props_std = eval_module_props.evaluate(global_step, train_env=envs, noise=False)

            # save stats
            if args.log_stats:
                for key, val in ppo_stats.items():
                    ppo_logs[key].append(ppo_stats[key])
                for key, val in props_stats.items():
                    props_logs[key].append(props_stats[key])

            np.savez(
                eval_module.log_path,
                timesteps=eval_module.evaluations_timesteps,
                returns=eval_module.evaluations_returns,
                successes=eval_module.evaluations_successes,
                **ppo_logs,
                **props_logs
            )

            if args.track:
                writer.add_scalar("charts/ppo_eval_return", target_ret, global_step)
                if args.props_eval:
                    writer.add_scalar("charts/props_eval_return", props_ret, global_step)

    envs.close()


if __name__ == "__main__":
    main()
