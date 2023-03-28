# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import pickle
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from gymnasium.wrappers.normalize import RunningMeanStd
from torch.distributions import Beta
from torch.distributions.normal import Normal

from evaluate import Evaluate
from utils import get_latest_run_id, NormalizeReward, NormalizeObservation


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="PPOROS", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Walker2d-v4", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="total timesteps of the experiments")
    parser.add_argument("--beta", type=int, default=False, help="Sample actions from Beta distribution rather than Gaussian")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4, help="the learning rate of the optimizer")
    parser.add_argument("--learning-rate-ros", "-lr-ros", type=float, default=1e-4, help="the learning rate of the ROS optimizer")
    parser.add_argument("--num-envs", type=int, default=1, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--buffer-history", "-b", type=int, default=4, help="Number of prior collect phases to store in buffer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--ent-coef-ros", type=float, default=0.01, help="coefficient of the entropy in ros update")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    parser.add_argument("--ros", type=float, default=True, help="True = use ROS policy to collect data, False = use target policy")
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
    # fmt: on

    if args.seed is None:
        args.seed = np.random.randint(2 ** 32 - 1)

    save_dir = f"{args.results_dir}/{args.env_id}/ppo/{args.results_subdir}"
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
            nn.Tanh(),
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

    def get_action(self, x):
        action_mean = self.actor_mean(x)
        return action_mean


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
        alpha = torch.exp(net_output[:, :self.action_dim])
        beta = torch.exp(net_output[:, self.action_dim:])
        probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x):
        net_output = self.net(x)
        alpha = torch.exp(net_output[:, :self.action_dim])
        beta = torch.exp(net_output[:, self.action_dim:])
        action_mean = alpha / (alpha+beta)
        return action_mean

def update_ppo():
    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

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

            loss = None
            if args.ros_target_kl is not None:
                if approx_kl > args.ros_target_kl:
                    # skipped_updates += 1
                    continue

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

        # if args.target_kl is not None:
        #     if approx_kl > args.target_kl:
        #         break

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
        writer.add_scalar("ppo/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("ppo/epochs", epoch + 1, global_step)
        writer.add_scalar("ppo/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("ppo/explained_variance", explained_var, global_step)
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


def normalize_obs(obs_rms, obs):
    """Normalises the observation using the running mean and variance of the observations."""
    print(obs_rms.mean)
    return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)


if __name__ == "__main__":
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
        writer = SummaryWriter(f"{run_name}")
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

    if args.beta:
        agent = AgentBeta(envs).to(device)
    else:
        agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.policy_path:
        agent = torch.load(args.policy_path)

    eval_module = Evaluate(model=agent, eval_env=None, n_eval_episodes=args.eval_episodes, log_path=args.save_dir, device=device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    obs_rms = RunningMeanStd(shape=envs.single_observation_space.shape)
    return_rms = RunningMeanStd(shape=())

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()

    timesteps = []
    sampling_error = []
    entropy_target = []
    entropy_ros = []

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

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

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

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

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        update_ppo()

        if update % args.eval_freq == 0:
            current_time = time.time() - start_time
            print(f"Training time: {int(current_time)} \tsteps per sec: {int(global_step / current_time)}")
            target_ret, target_std = eval_module.evaluate(global_step, train_env=envs)
            if args.track:
                writer.add_scalar("charts/ppo_eval_return", target_ret, global_step)

            if args.compute_sampling_error:
                agent_mle = Agent(envs).to(device)
                optimizer_mle = optim.Adam(agent_mle.parameters(), lr=1e-3)

                b_obs = obs.reshape(-1, obs_dim)
                b_actions = actions.reshape(-1, action_dim)

                for i in range(10000):
                    _, logprobs_mle, _, _ = agent_mle.get_action_and_value(b_obs, b_actions)
                    loss = -torch.mean(logprobs_mle)

                    # if i % 100 == 0:
                    #     print(i, loss.item())

                    optimizer_mle.zero_grad()
                    loss.backward()
                    optimizer_mle.step()

                with torch.no_grad():

                    _, logprobs_target, ent_target, _ = agent.get_action_and_value(b_obs, b_actions)
                    logratio = logprobs_mle - logprobs
                    ratio = logratio.exp()
                    approx_kl_mle_target = ((ratio - 1) - logratio).mean()
                    print('D_kl( mle || target ) = ', approx_kl_mle_target.item())

                    sampling_error.append(approx_kl_mle_target.item())
                    entropy_target.append(ent_target.mean().item())
                    timesteps.append(global_step)

                    np.savez(f'{args.save_dir}/stats.npz',
                             timesteps=timesteps,
                             sampling_error=sampling_error,
                             entropy_target=entropy_target)

    envs.close()
    # writer.close()