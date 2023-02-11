# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import copy
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from PPOROS.evaluate import Evaluate
from PPOROS.utils import get_latest_run_id


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=None, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v4", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=50000, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1024, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=1, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    print(args)
    # fmt: on
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
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
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
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x, deterministic=True):
        action_mean = self.actor_mean(x)
        # action_logstd = self.actor_logstd.expand_as(action_mean)
        # action_std = torch.exp(action_logstd)
        # probs = Normal(action_mean, action_std)
        return action_mean

def log_statistics(writer, v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var):
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    # print("SPS:", int(global_step / (time.time() - start_time)))
    # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.track and args.capture_video:
        for filename in os.listdir(f"videos/{run_name}"):
            if filename not in video_filenames and filename.endswith(".mp4"):
                wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                video_filenames.add(filename)

def update_ppo(args, obs, logprobs, actions, advantages, returns, values):

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

    return v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var


def update_ros(args, agent_ros, envs, optimizer_ros, obs, logprobs, actions):
    # update historic values and logprobs
    # with torch.no_grad():
    #     # for j in range(2):
    #     #     # if j == buffer_epoch_index: continue
    #     _, new_logprob, _, new_value = agent_ros.get_action_and_value(obs.reshape(-1, 4), actions.reshape(-1, 1))
    #     logprobs = new_logprob.reshape(-1, 1)

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
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
            loss = pg_loss - 0.01 * entropy_loss

            optimizer_ros.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent_ros.parameters(), args.max_grad_norm)
            optimizer_ros.step()

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                break


    return pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs


if __name__ == "__main__":
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(2**32-1)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True, no longer works for gymnasium
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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    agent_ros = copy.deepcopy(agent)  # initialize ros policy to be equal to the eval policy
    optimizer_ros = optim.Adam(agent_ros.parameters(), lr=args.learning_rate, eps=1e-5)

    save_dir = f"results/{args.env_id}/ppo_ros"
    run_id = get_latest_run_id(save_dir=save_dir) + 1
    save_dir += f"/run_{run_id}"

    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    eval_module = Evaluate(model=agent, eval_env=eval_envs, n_eval_episodes=10, log_path=save_dir, device=device)

    # Save config
    with open(os.path.join(save_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    history_k = 2

    # ALGO Logic: Storage setup
    obs = torch.zeros((history_k, args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    obs_p = torch.zeros((history_k, args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((history_k, args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((history_k, args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((history_k, args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((history_k, args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((history_k, args.num_steps, args.num_envs)).to(device)
    returns = torch.zeros((history_k, args.num_steps, args.num_envs)).to(device)
    advantages = torch.zeros((history_k, args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    global_epoch = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()

    obs_dim = 4
    action_dim = 1

    # num_updates = 4
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        # if args.anneal_lr:
        #     frac = 1.0 - (update - 1.0) / num_updates
        #     lrnow = frac * args.learning_rate
        #     optimizer.param_groups[0]["lr"] = lrnow

        global_epoch += 1
        buffer_epoch_index = (global_epoch - 1) % history_k

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[buffer_epoch_index, step] = next_obs
            dones[buffer_epoch_index, step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # action, logprob, _, value = agent.get_action_and_value(next_obs)
                action, _, _, value = agent_ros.get_action_and_value(next_obs)
                _, logprob, _, _ = agent.get_action_and_value(next_obs, action)
                values[buffer_epoch_index, step] = value.flatten()
            actions[buffer_epoch_index, step] = action
            logprobs[buffer_epoch_index, step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[buffer_epoch_index, step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)


        # update historic values and logprobs
        with torch.no_grad():
            for j in range(history_k):
                # if j == buffer_epoch_index: continue
                _, new_logprob, _, new_value = agent.get_action_and_value(obs[j].reshape(-1, obs_dim), actions[j].reshape(-1, action_dim))
                values[j] = new_value
                logprobs[j] = new_logprob.reshape(-1,1)

        num_rshifts = (history_k - 1) - buffer_epoch_index
        # # bootstrap value if not done
        with torch.no_grad():
            lastgaelam = 0
            if update > history_k:
                epoch_rewards = torch.roll(rewards, num_rshifts, dims=0).reshape(-1, 1)
                epoch_values = torch.roll(values, num_rshifts, dims=0).reshape(-1, 1)
                epoch_dones = torch.roll(dones, num_rshifts, dims=0).reshape(-1, 1)
            else:
                epoch_rewards = rewards[:update].reshape(-1, 1)
                epoch_values = values[:update].reshape(-1,1)
                epoch_dones = dones[:update].reshape(-1,1)

            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(epoch_rewards).to(device)
            next_done = next_done

            num_steps = epoch_rewards.shape[0]
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - epoch_dones[t + 1]
                    nextvalues = epoch_values[t + 1]
                delta = epoch_rewards[t] + args.gamma * nextvalues * nextnonterminal - epoch_values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + epoch_values

        if update >= history_k:
            returns = returns.reshape(history_k, -1, 1)
            advantages = advantages.reshape(history_k, -1, 1)
        else:
            returns = returns.reshape(buffer_epoch_index+1, -1, 1)
            advantages = advantages.reshape(buffer_epoch_index+1, -1, 1)

        # bootstrap value if not done
        # with torch.no_grad():
        #     lastgaelam = 0
        #     for j in range(history_k):
        #         epoch_j = (buffer_epoch_index - j) % history_k
        #         epoch_rewards = rewards[epoch_j]
        #         epoch_values = values[epoch_j]
        #         epoch_dones = dones[epoch_j]
        #
        #         advantages = torch.zeros_like(epoch_rewards).to(device)
        #         if epoch_j == buffer_epoch_index:
        #             next_value = agent.get_value(next_obs).reshape(1, -1)
        #             next_done = next_done
        #         else:
        #             next_value = agent.get_value(obs[(epoch_j + 1) % history_k, 0]).reshape(1, -1)
        #             next_done = dones[(epoch_j + 1) % history_k, 0]
        #
        #         for t in reversed(range(args.num_steps)):
        #             if t == args.num_steps - 1:
        #                 nextnonterminal = 1.0 - next_done
        #                 nextvalues = next_value
        #             else:
        #                 nextnonterminal = 1.0 - epoch_dones[t + 1]
        #                 nextvalues = epoch_values[t + 1]
        #             delta = epoch_rewards[t] + args.gamma * nextvalues * nextnonterminal - epoch_values[t]
        #             advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        #         returns[epoch_j] = advantages + epoch_values

        if update % 1 == 0:
            if update < history_k:
                # args.batch_size = 2048*update
                v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var = \
                    update_ppo(args, obs[:global_epoch], logprobs[:global_epoch], actions[:global_epoch], advantages[:global_epoch], returns[:global_epoch], values[:global_epoch])

                pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs = \
                    update_ros(args, agent_ros, envs, optimizer_ros, obs[:global_epoch], logprobs[:global_epoch], actions[:global_epoch])
            else:
                # args.batch_size = 2048*history_k
                v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var = \
                    update_ppo(args, obs, logprobs, actions, advantages, returns, values)

                pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs = \
                    update_ros(args, agent_ros, envs, optimizer_ros, obs, logprobs, actions)

            log_statistics(writer, v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var)
        # if update % 1 == 0:
        #     current_time = time.time()-start_time
        #     print(f"Training time: {int(current_time)} \tsteps per sec: {int(global_step / current_time)}")
        #     eval_module.evaluate(global_step)


        # log_statistics_ros(global_step, writer, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs)

    envs.close()
    writer.close()