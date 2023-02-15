# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import copy
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions.categorical import Categorical

from PPOROS.evaluate import Evaluate
from PPOROS.utils import get_latest_run_id, parse_args, make_env, Agent


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
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

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

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

def update_ros(agent_ros, envs, optimizer_ros, obs, logprobs, actions, global_step, args, buffer_size, writer):

    # flatten the batch
    if global_step < buffer_size:
        end = global_step
    else:
        end = -args.num_steps

    # flatten the batch
    b_obs = obs[:end].reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs[:end].reshape(-1)
    b_actions = actions[:end].reshape((-1,) + envs.single_action_space.shape)

    # Optimizing the policy and value network
    batch_size = b_obs.shape[0]
    minibatch_size = int(batch_size // args.num_minibatches)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, _ = agent_ros.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            # Policy loss
            pg_loss1 = -ratio
            pg_loss2 = -torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef_ros * entropy_loss

            optimizer_ros.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent_ros.parameters(), args.max_grad_norm)
            optimizer_ros.step()

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                break

def main():
    args = parse_args()
    writer = None

    print(f'Results will be saved to {args.save_dir}')

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, args.save_dir) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # PPO target agent
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ROS behavior agent
    agent_ros = copy.deepcopy(agent)  # initialize ros policy to be equal to the eval policy
    optimizer_ros = optim.Adam(agent_ros.parameters(), lr=args.learning_rate_ros, eps=1e-5)

    # Evaluation modules
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, args.save_dir) for i in range(args.num_envs)]
    )
    eval_envs_ros = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, args.save_dir) for i in range(args.num_envs)]
    )
    eval_module = Evaluate(model=agent, eval_env=eval_envs, n_eval_episodes=args.eval_episodes, log_path=args.save_dir, device=device)
    eval_module_ros = Evaluate(model=agent_ros, eval_env=eval_envs_ros, n_eval_episodes=args.eval_episodes, log_path=args.save_dir, device=device, suffix='ros')

    # ALGO Logic: Storage setup
    history_k = args.buffer_history
    buffer_size = history_k * args.num_steps

    # ALGO Logic: Storage setup
    obs_buffer = torch.zeros((buffer_size, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buffer = torch.zeros((buffer_size, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards_buffer = torch.zeros((buffer_size, args.num_envs)).to(device)
    dones_buffer = torch.zeros((buffer_size, args.num_envs)).to(device)

    buffer_pos = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = 1

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs_buffer[buffer_pos] = next_obs
            dones_buffer[buffer_pos] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.ros:
                    action, _, _, _ = agent_ros.get_action_and_value(next_obs)
                else:
                    action, _, _, _ = agent.get_action_and_value(next_obs)
                actions_buffer[buffer_pos] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards_buffer[buffer_pos] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            buffer_pos += 1
            buffer_pos %= buffer_size

        # Reorder the replay data so that the most new data is at the back.
        # We do this to avoid modifying existing the CleanRL PPO update code, which assumes buffer[0] is the oldest transition
        # We copy the replay data for clarity. There's a more efficient way to do.
        if global_step < buffer_size:
            indices = np.arange(buffer_pos)
        else:
            indices = (np.arange(buffer_size) + buffer_pos) % buffer_size

        obs = obs_buffer.clone()[indices]
        actions = actions_buffer.clone()[indices]
        rewards = rewards_buffer.clone()[indices]
        dones = dones_buffer.clone()[indices]

        # obs_buffer = obs_buffer[indices]
        # actions_buffer = actions_buffer[indices]
        # rewards_buffer = rewards_buffer[indices]
        # dones_buffer = dones_buffer[indices]

        # Compute value estimates and logprobs
        with torch.no_grad():
            _, new_logprob, _, new_value = agent.get_action_and_value(obs.reshape(-1, obs_dim), actions.reshape(-1))
            values = new_value
            logprobs = new_logprob.reshape(-1, 1)

        # Compute returns and advantages -- bootstrap value if not done
        with torch.no_grad():
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

        # PPO target update
        update_ppo(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer)

        if args.ros:
            # Set ROS policy equal to current target policy
            for source_param, dump_param in zip(agent_ros.parameters(), agent.parameters()):
                source_param.data.copy_(dump_param.data)

            # ROS behavior update
            update_ros(agent_ros, envs, optimizer_ros, obs, logprobs, actions, global_step, args, buffer_size, writer)

        # indices_invert = np.empty_like(indices)
        # indices_invert[indices] = np.arange(indices_invert)
        # obs = obs_buffer[indices_invert]
        # actions = actions_buffer[indices_invert]
        # rewards = rewards_buffer[indices_invert]
        # dones = dones_buffer[indices_invert]


        if update % args.eval_freq == 0:
            current_time = time.time() - start_time
            print(f"Training time: {int(current_time)} \tsteps per sec: {int(global_step / current_time)}")
            eval_module.evaluate_old_gym(global_step)
            eval_module_ros.evaluate_old_gym(global_step)


    envs.close()

if __name__ == "__main__":
    main()