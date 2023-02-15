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
from gymnasium.wrappers.normalize import RunningMeanStd
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from PPOROS.evaluate import Evaluate
from PPOROS.utils import get_latest_run_id, parse_args


class Evaluate:
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
            self,
            model,
            eval_env,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: str = None,
            suffix: str = '',
            best_model_save_path: str = None,
            deterministic: bool = True,
            device=None,
    ):
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.device = device

        self.model = model
        self.eval_env = eval_env
        self.best_model_save_path = log_path
        self.suffix = suffix

        # Logs will be written in ``evaluations.npz``
        os.makedirs(name=log_path, exist_ok=True)
        if log_path is not None:
            if self.suffix != '':
                self.log_path = os.path.join(log_path, f"evaluations_{suffix}")
            else:
                self.log_path = os.path.join(log_path, f"evaluations")
        self.evaluations_returns = []
        self.evaluations_timesteps = []
        self.evaluations_successes = []
        # For computing success rate
        self._is_success_buffer = []

    def evaluate(self, t, train_envs=None):
        # if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
        returns, successes = self._evaluate(train_envs)

        if self.log_path is not None:
            self.evaluations_timesteps.append(t)
            self.evaluations_returns.append(returns)
            self.evaluations_successes.append(successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                returns=self.evaluations_returns,
                successes=self.evaluations_successes,
            )

            mean_reward, std_reward = np.mean(returns), np.std(returns)
            mean_success, std_success = np.mean(successes), np.std(successes)

            self.last_mean_reward = mean_reward

            print(f"Eval num_timesteps={t}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            # print(f"Eval num_timesteps={t}, " f"episode_success={mean_success:.2f} +/- {std_success:.2f}")

            if mean_reward > self.best_mean_reward:
                print("New best mean reward!")
                torch.save(self.model, os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward

    def _evaluate(self, train_envs):
        eval_returns = []
        eval_successes = []

        obs, _ = self.eval_env.reset()
        for episode_i in range(self.n_eval_episodes):
            ep_returns = []
            ep_successes = []
            done = False
            step = 0
            while not done:
                step += 1
                # ALGO LOGIC: put action logic here
                with torch.no_grad():
                    if train_envs:
                        obs = (obs - train_envs.obs_rms.mean) / np.sqrt(train_envs.obs_rms.var + 1e-8)
                    actions = self.model.get_action(torch.Tensor(obs).to(self.device))
                    # actions = self.model(torch.Tensor(obs).to(self.device))
                    actions = actions.cpu().numpy().clip(self.eval_env.single_action_space.low,
                                                         self.eval_env.single_action_space.high)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminateds, truncateds, infos = self.eval_env.step(actions)
                done = terminateds[0] or truncateds[0]

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

                ep_returns.append(rewards[0])
                ep_successes.append(terminateds[0])

            eval_returns.append(np.sum(ep_returns))
            eval_successes.append(np.sum(ep_successes) * 100)

        return eval_returns, eval_successes
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

def make_eval_env(env_id, idx, capture_video, run_name, gamma):
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
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    save_dir = f"results/{args.env_id}/ppo"
    run_id = get_latest_run_id(save_dir=save_dir) + 1
    save_dir += f"/run_{run_id}"


    # Save config
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

        # ROS behavior agent
    agent_ros = copy.deepcopy(agent)  # initialize ros policy to be equal to the eval policy
    optimizer_ros = optim.Adam(agent_ros.parameters(), lr=args.learning_rate_ros, eps=1e-5)

    # Evaluation modules
    eval_envs = gym.vector.SyncVectorEnv(
        [make_eval_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    eval_envs_ros = gym.vector.SyncVectorEnv(
        [make_eval_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    eval_module = Evaluate(model=agent, eval_env=eval_envs, n_eval_episodes=args.eval_episodes, log_path=args.save_dir,
                           device=device)
    eval_module_ros = Evaluate(model=agent_ros, eval_env=eval_envs_ros, n_eval_episodes=args.eval_episodes,
                               log_path=args.save_dir, device=device, suffix='ros')

    # ALGO Logic: Storage setup
    history_k = args.buffer_history
    history_k = 1
    buffer_size = history_k * args.num_steps
    # ALGO Logic: Storage setup
    # ALGO Logic: Storage setup
    obs_buffer = torch.zeros((buffer_size, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_buffer = torch.zeros((buffer_size, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards_buffer = torch.zeros((buffer_size, args.num_envs)).to(device)
    dones_buffer = torch.zeros((buffer_size, args.num_envs)).to(device)

    buffer_pos = 0

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    # num_updates = 2
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        # if args.anneal_lr:
        #     frac = 1.0 - (update - 1.0) / num_updates
        #     lrnow = frac * args.learning_rate
        #     optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions_buffer[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards_buffer[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        if global_step < buffer_size:
            indices = np.arange(buffer_pos)
        else:
            indices = (np.arange(buffer_size) + buffer_pos) % buffer_size

        obs = obs_buffer.clone()[indices]
        actions = actions_buffer.clone()[indices]
        rewards = rewards_buffer.clone()[indices]
        dones = dones_buffer.clone()[indices]

        with torch.no_grad():
            _, new_logprob, _, new_value = agent.get_action_and_value(obs.reshape(-1, obs_dim), actions.reshape(-1, action_dim))
            values = new_value
            logprobs = new_logprob.reshape(-1, 1)

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

        update_ppo(args, obs, logprobs, actions, advantages, returns, values)

        if update % 1 == 0:
            current_time = time.time()-start_time
            print(f"Training time: {int(current_time)} \tsteps per sec: {int(global_step / current_time)}")
            eval_module.evaluate(global_step, train_envs=envs.envs[0])

    envs.close()
    writer.close()
