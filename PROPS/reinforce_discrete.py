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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from PROPS.utils import Evaluate, AgentDiscrete, EvaluateDiscrete, ConfigLoader
from PROPS.utils import get_latest_run_id, make_env, Agent

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    # We use integers 0/1 instead of booleans False/True simply because the server we use for all experiments may
    # interpret False/True as strings instead of booleans.

    # weights and biases (wandb) parameters. Wandb is disabled by default.
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="If toggled, this experiment will be tracked with Weights and Biases (wandb)")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="Wandb experiment name")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="Wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb project entity (team)")
    parser.add_argument("--wandb-login-key", type=str, default=None, help="Wandb login key")

    # Saving and logging parameters
    parser.add_argument("--log-stats", type=int, default=1, help="If true, training statistics are logged")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluate PPO and/or PROPS policy every eval_freq PPO updates")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes over which policies are evaluated")
    parser.add_argument("--results-dir", "-f", type=str, default="results", help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")
    parser.add_argument("--results-subdir", "-s", type=str, default="", help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")
    parser.add_argument("--run-id", type=int, default=None, help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")

    # General training parameters (both PROPS and PPO)
    parser.add_argument("--env-id", type=str, default="GridWorld-5x5-v0", help="Environment id")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=500000, help="Number of timesteps to train")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="If toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="If toggled, cuda will be enabled by default")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    # PPO hyperparameters
    parser.add_argument("--num-steps", type=int, default=10000, help="PPO target batch size (n in paper), the number of steps to collect between each PPO policy update")
    parser.add_argument("--num-traj", type=int, default=10, help="PPO target batch size, the number of trajectories to collect between each PPO policy update")
    parser.add_argument("--buffer-batches", "-b", type=int, default=100, help="Number of PPO target batches to store in the replay buffer (b in paper)")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="PPO Adam optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=1, help="Discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=1, help="General advantage estimation lambda (not the lambda used for PROPS")
    parser.add_argument("--num-minibatches", type=int, default=32, help="Number of minibatches updates for PPO update")
    parser.add_argument("--update-epochs", type=int, default=10, help="Number of epochs for PPO update")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization for PPO update")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="Surrogate clipping coefficient \epsilon for PPO update")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy loss coefficient for PPO update")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient for PPO update")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum norm for gradient clipping for PPO update")
    parser.add_argument("--target-kl", type=float, default=0.03, help="Target/cutoff KL divergence threshold for PPO update")
    parser.add_argument("--linear", type=int, default=0, help="")
    parser.add_argument("--actor-critic", type=int, default=0, help="")
    parser.add_argument("--reinforce", type=int, default=0, help="")
    parser.add_argument("--oracle-adaptive", type=int, default=0, help="")
    parser.add_argument("--random-sampling", type=int, default=0, help="")

    # PROPS/ROS hyperparameters
    parser.add_argument("--props", type=int, default=0, help="If True, use PROPS to collect data, otherwise use on-policy sampling")
    parser.add_argument("--ros", type=int, default=0, help="If True, use ROS to collect data, otherwise use on-policy sampling")
    parser.add_argument("--props-num-steps", type=int, default=1024, help="PROPS behavior batch size (m in paper), the number of steps to run in each environment per policy rollout")
    parser.add_argument("--props-num-traj", type=int, default=1, help="PROPS behavior batch size (m in paper), the number of steps to run in each environment per policy rollout")
    parser.add_argument("--props-learning-rate", "-props-lr", type=float, default=1e-3, help="PROPS Adam optimizer learning rate")
    parser.add_argument("--props-anneal-lr", type=lambda x: bool(strtobool(x)), default=0, nargs="?", const=False, help="Toggle learning rate annealing for PROPS policy")
    parser.add_argument("--props-clip-coef", type=float, default=0.1, help="Surrogate clipping coefficient \epsilon_PROPS for PROPS")
    parser.add_argument("--props-max-grad-norm", type=float, default=0.5, help="Maximum norm for gradient clipping for PROPS update")
    parser.add_argument("--props-num-minibatches", type=int, default=1, help="Number of minibatches updates for PROPS update")
    parser.add_argument("--props-update-epochs", type=int, default=4, help="Number of epochs for PROPS update")
    parser.add_argument("--props-target-kl", type=float, default=0.01, help="Target/cutoff KL divergence threshold for PROPS update")
    parser.add_argument("--props-lambda", type=float, default=0.1, help="Regularization coefficient for PROPS update")
    parser.add_argument("--props-adv", type=int, default=False, help="If True, the PROPS update is weighted using the absolute advantage |A(s,a)|")
    parser.add_argument("--props-eval", type=int, default=False, help="If set, the PROPS policy is evaluated every props_eval ")

    # Sampling error (se)
    parser.add_argument("--se", type=int, default=0, help="If True, sampling error is computed every se_freq PPO updates.")
    parser.add_argument("--se-ref", type=int, default=1, help="If True, on-policy sampling error is computed using the PPO policy sequence obtained while using PROPS. Only applies if se is True.")
    parser.add_argument("--se-lr", type=float, default=1e-3, help="Adam optimizer learning rate used to compute the empirical (maximum likelihood) policy in sampling error computation.")
    parser.add_argument("--se-epochs", type=int, default=250, help="Number of epochs to compute empirical (maximum likelihood) policy.")
    parser.add_argument("--se-freq", type=int, default=None, help="Compute sampling error very se_freq PPO updates")
    parser.add_argument("--se-debug", type=int, default=None, help="Only run PROPS when we evaluate sampling error")

    # loading pretrained models
    parser.add_argument("--policy-path", type=str, default=None, help="Path of pretrained policy to load")
    parser.add_argument("--normalization-dir", type=str, default=None, help="Directory contatining normalization statistics of pretrained policy")



    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.buffer_size = args.buffer_batches * args.batch_size
    args.minibatch_size = int(args.buffer_size // args.num_minibatches)
    args.props_minibatch_size = int((args.buffer_size - args.props_num_steps) // args.props_num_minibatches)
    # args.eval_freq = args.num_steps

    # cuda support. Currently does not work with normalization
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.seed is None:
        if args.run_id:
            args.seed = np.random.randint(args.run_id)
        else:
            args.seed = np.random.randint(2 ** 32 - 1)

    # set save_dir
    assert not (args.reinforce == 1 and args.actor_critic == 1)
    assert not (args.props == 1 and args.ros == 1)


    algo = 'reinforce'
    args.update_epochs = 1
    args.minibatch_size = args.buffer_size
    args.ent_coef = 0
    args.buffer_size = int(args.num_traj * args.num_steps)
    args.gae_lambda = 1 # advantage estimates reduce to MC discounted return estimates
    if args.props:
        sampling = 'props'
    elif args.ros:
        sampling = 'ros'
        args.minibatch_size = args.buffer_size
        args.props_update_epochs = 1
        args.props_clip_coef = 9999999
        args.props_target_kl = 9999999
        args.props_lambda = 0
    else:
        sampling = 'on_policy'
        args.props_num_steps = args.num_steps # to force num_props_updates = num_updates

    if args.oracle_adaptive:
        args.algo = 'oracle_adaptive'
    else:
        args.algo = f'{algo}_{sampling}'

    args.save_dir = f"{args.results_dir}/{args.env_id}/{args.algo}/{args.results_subdir}"


    if args.config:
        with open(args.config) as f:
            try:
                # args = yaml.load(f, Loader=ConfigLoader)
                args_loaded = yaml.unsafe_load(f)
                # # otherwise we use the same run_id and seed for every experiment
                args_loaded.seed = args.seed
                args_loaded.run_id = args.run_id
                args_loaded.save_dir = f"{args_loaded.results_dir}/{args_loaded.env_id}/{args_loaded.algo}/{args_loaded.results_subdir}"

                # if args_loaded.run_id is not None:
                #     args_loaded.save_dir += f"/run_{args.run_id}"
                # else:
                #     run_id = get_latest_run_id(save_dir=save_dir) + 1
                #     args_loaded.save_dir += f"/run_{run_id}"

                args = args_loaded
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)

    if args.run_id is not None:
        args.save_dir += f"/run_{args.run_id}"
    else:
        run_id = get_latest_run_id(save_dir=args.save_dir) + 1
        args.save_dir += f"/run_{run_id}"

    # dump training config to save dir
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    return args


def update_reinforce(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer):
    # PPO UPDATE

    # flatten buffer data
    b_obs = obs.view((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.view(-1)
    b_actions = actions.view((-1,) + envs.single_action_space.shape).long()
    b_advantages = advantages.view(-1)
    b_returns = returns.view(-1)
    b_values = values.view(-1)

    _, _, newlogprobs, entropy, newvalues = agent.get_action_and_value(b_obs, b_actions)
    newvalues = newvalues.view(-1)

    # pg_loss = -(b_returns.mean() * newlogprobs).mean()
    pg_loss = -(b_advantages * newlogprobs).mean()

    v_loss = 0.5 * ((newvalues - b_advantages) ** 2).mean()
    entropy_loss = entropy.mean()

    loss = pg_loss + v_loss * args.vf_coef # - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    optimizer.zero_grad()
    loss.backward()

    # grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    grad = [p.grad.reshape(-1) for p in agent.parameters() if p.grad is not None and p.requires_grad]
    grad = torch.concat(grad)

    ppo_stats = {
        # 't': global_step,
        # 'ppo_value_loss': float(v_loss.item()),
        'ppo_policy_loss': float(pg_loss.item()),
        'ppo_entropy_loss': float(entropy_loss.item()),
    }
    if args.track:
        writer.add_scalar("ppo/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("ppo/value_loss", v_loss.item(), global_step)
        writer.add_scalar("ppo/policy_loss", pg_loss.item(), global_step)

    return ppo_stats, grad

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

    # PPO target agent
    agent = AgentDiscrete(envs, linear=args.linear).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # optimizer = optim.SGD(agent.parameters(), lr=args.learning_rate)

    # load pretrained policy and normalization information
    if args.policy_path:
        agent = torch.load(args.policy_path).to(args.device)

    # ROS behavior agent
    agent_props = copy.deepcopy(agent).to(args.device)  # initialize props policy to be equal to the eval policy
    props_optimizer = optim.Adam(agent_props.parameters(), lr=args.props_learning_rate, eps=1e-5)

    # Evaluation modules
    eval_module = EvaluateDiscrete(model=agent, eval_env=None, n_eval_episodes=args.eval_episodes, log_path=args.save_dir,
                           device=args.device)
    eval_module_props = EvaluateDiscrete(model=agent_props, eval_env=None, n_eval_episodes=args.eval_episodes,
                                 log_path=args.save_dir, device=args.device, suffix='props')

    args.buffer_size = envs.envs[0].spec.max_episode_steps * args.num_traj * args.buffer_batches
    obs_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    actions_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_action_space.shape).to(args.device)
    rewards_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(args.device)
    dones_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(args.device)
    values_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(args.device)
    returns_buffer = torch.zeros((args.num_traj * args.buffer_batches, args.num_envs)).to(args.device)
    discounted_return = torch.zeros((args.num_envs,)).to(args.device).type(torch.float)
    buffer_pos = 0  # index of buffer position to be updated in the current timestep
    traj_count = 0
    return_buffer_pos = 0

    # initialize RL loop
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    next_terminated = torch.zeros(args.num_envs).to(args.device)

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
    times = []
    updates = [0]
    timesteps = [0]
    all_grad_accuracy = []

    start_time = time.time()
    global_step = 0
    target_update = 0
    props_update = 0

    # evaluate initial policy
    eval_returns, eval_obs, eval_actions, eval_rewards, sa_eval = eval_module.simulate(train_env=envs)
    eval_obs = torch.from_numpy(eval_obs).view((-1,) + envs.single_observation_space.shape)
    eval_actions = torch.from_numpy(eval_actions).view((-1,) + envs.single_action_space.shape).long()
    eval_returns = torch.from_numpy(eval_returns).view(-1)

    _, _, newlogprobs, entropy, newvalues = agent.get_action_and_value(eval_obs, eval_actions)
    pg_loss = -(eval_returns.mean() * newlogprobs).mean()
    loss = pg_loss #+ v_loss * args.vf_coef # - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    optimizer.zero_grad()
    loss.backward()
    grad_true = [p.grad.reshape(-1) for p in agent.parameters() if p.grad is not None and p.requires_grad]
    grad_true = torch.concat(grad_true)

    ns_eval = sa_eval.sum()
    sa_true = sa_eval/ns_eval

    if args.props_eval:
        eval_module_props.evaluate(global_step, train_env=envs, noise=False)

    ppo_stats = {}
    props_stats = {}

    all_se = []
    all_pi = []
    all_sa_counts = []
    sa_counts = np.zeros(shape=(envs.single_observation_space.shape[-1], envs.single_action_space.n))
    possible_actions = np.arange(envs.single_action_space.n)

    episode_t = 0
    for global_step in range(args.total_timesteps):
        # collect a transition
        global_step += 1 * args.num_envs
        episode_t += 1 * args.num_envs
        obs_buffer[buffer_pos] = next_obs # store unnormalized obs
        dones_buffer[buffer_pos] = next_done

        with torch.no_grad():
            if args.oracle_adaptive:
                s_idx = np.argmax(next_obs)
                sa = sa_counts[s_idx]
                # never_sampled_mask = (sa == 0)

                pi = agent.get_pi_s(next_obs)[0]

                if np.sum(sa) == 0:
                    a_idx = np.random.choice(possible_actions, p=pi)
                else:
                    pi_empirical = sa / np.sum(sa)
                    a_idx = np.argmin(pi_empirical - agent.get_pi_s(next_obs))

                #
                # if np.any(never_sampled_mask):
                #     pi[~never_sampled_mask] = 0
                #     pi /= np.sum(pi[never_sampled_mask])
                #     # print(np.sum(pi[never_sampled_mask]))
                #
                #     a_idx = np.random.choice(possible_actions, p=pi)
                # else:
                #     pi_empirical = sa / np.sum(sa)
                #
                #     a_idx = np.argmin(pi_empirical - agent.get_pi_s(next_obs))
                # action = np.zeros(envs.single_action_space.n)
                action = torch.Tensor([a_idx])

                # values = torch.zeros()
                # logprobs = 0
                _, _, logprobs, _, values = agent_props.get_action_and_value(next_obs, action)
            elif args.props:
                action, action_probs, logprobs, entropy, values = agent_props.get_action_and_value(next_obs)
                # fetch value and logprob w.r.t target policy (not behavior policy)
                _, _, logprobs, _, values = agent_props.get_action_and_value(next_obs, action)

                a_idx = action[0]
            else:
                action, action_probs, logprobs, entropy, values = agent.get_action_and_value(next_obs)
                a_idx = action[0]
            actions_buffer[buffer_pos] = action
            values_buffer[buffer_pos] = values

        s_idx = np.where(next_obs[0] == 1)[0][0]
        sa_counts[s_idx, a_idx] += 1

        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        next_obs, next_done, next_terminated = torch.Tensor(next_obs).to(args.device), torch.Tensor(terminated | truncated).to(args.device), torch.Tensor(next_terminated).to(args.device)

        rewards_buffer[buffer_pos] = torch.tensor(reward).to(args.device).view(-1)
        dones_buffer[buffer_pos] = next_done
        discounted_return += args.gamma**episode_t * reward
        for info in infos.get("final_info", []):
            # Skip the envs that are not done
            if info is None:
                continue

            if args.track:
                writer.add_scalar("charts/props_train_ret", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episode_length", info["episode"]["l"], global_step)

        buffer_pos += 1
        buffer_pos %= args.buffer_size

        # determine what all needs to be done at this timestep
        do_ppo_update = False
        if next_done:
            returns_buffer[return_buffer_pos] = discounted_return
            return_buffer_pos += 1
            return_buffer_pos %= (args.num_traj * args.buffer_batches)

            discounted_return[:] = 0
            episode_t = 0
            traj_count += 1
        if traj_count == args.num_traj:
            do_ppo_update = True
        if traj_count == (args.num_traj * args.buffer_batches):
            traj_count = 0
            sa_counts[:, :] = 0

        # do_props_update = args.props and global_step % args.props_num_steps == 0
        do_props_update = args.props and (terminated or truncated)
        do_eval = (global_step + 1) % args.eval_freq == 0

        if do_ppo_update or do_props_update:
            if buffer_pos == 0:
                obs = obs_buffer
                actions = actions_buffer
                rewards = rewards_buffer
                dones = dones_buffer
                returns = returns_buffer
                values = values_buffer
            else:
                obs = obs_buffer[:buffer_pos]
                actions = actions_buffer[:buffer_pos]
                rewards = rewards_buffer[:buffer_pos]
                dones = dones_buffer[:buffer_pos]
                returns = returns_buffer[:buffer_pos]
                values = values_buffer[:buffer_pos]
            # else:
            #     obs = obs_buffer
            #     actions = actions_buffer
            #     rewards = rewards_buffer
            #     dones = dones_buffer
            #     returns = returns_buffer
            #     values = values_buffer

            # Store the b previous target policies. We do this so we can compute on-policy sampling error with respect to
            # the target policy sequence obtained by PROPS.
            if global_step % args.num_steps == 0:
                next_obs_buffer.append(copy.deepcopy(next_obs))
                agent_buffer.append(copy.deepcopy(agent))

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(args.device)
                lastgaelam = 0
                num_steps = len(obs)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_terminated
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                # returns = advantages + values #@TODO

            # PPO update
            if do_ppo_update:
                target_update += 1

                # ppo_stats = update_ppo(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer)
                ppo_stats, grad_empirical = update_reinforce(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer)
                    # print(global_step)

        # Evaluate agent performance
        if do_eval:
            current_time = time.time() - start_time
            print(f"Training time: {int(current_time)} \tsteps per sec: {int(global_step / current_time)}")
            agent = agent.to(args.device)
            agent_props = agent_props.to(args.device)
            # # Evaluate PPO policy
            target_ret, target_std, sa_eval = eval_module.evaluate(global_step, train_env=envs, noise=False)

            # if buffer_pos == 0:
            #     obs = obs_buffer
            #     actions = actions_buffer
            #     rewards = rewards_buffer
            #     dones = dones_buffer
            #     returns = returns_buffer
            #     values = values_buffer
            # else:
            #     obs = obs_buffer[:buffer_pos]
            #     actions = actions_buffer[:buffer_pos]
            #     rewards = rewards_buffer[:buffer_pos]
            #     dones = dones_buffer[:buffer_pos]
            #     returns = returns_buffer[:buffer_pos]
            #     values = values_buffer[:buffer_pos]
            #
            # advantages = None
            # ppo_stats, grad_empirical = update_reinforce(agent, optimizer, envs, obs, logprobs, actions, advantages,
            #                                              returns, values, args, global_step, writer)
            #
            # grad_empirical = [p.grad.reshape(-1) for p in agent.parameters() if p.grad is not None and p.requires_grad]
            # grad_empirical = torch.concat(grad_empirical)
            #
            # grad_accuracy = torch.dot(grad_empirical, grad_true).detach().numpy()/torch.norm(grad_empirical)/torch.norm(grad_true)
            # print(torch.norm(grad_empirical))
            # all_grad_accuracy.append(grad_accuracy.item())
            # # print(all_grad_accuracy)

            ns = sa_counts.sum()
            # print(sa_counts, sa_counts.sum())
            # print(sa_counts/ns - sa_eval/ns_eval)
            se = np.abs(sa_counts/ns - sa_true).sum()
            all_se.append(se)
            # print(all_se)

            # save stats
            if args.log_stats:
                for key, val in ppo_stats.items():
                    ppo_logs[key].append(ppo_stats[key])
                for key, val in props_stats.items():
                    props_logs[key].append(props_stats[key])
            times.append(current_time)
            timesteps.append(global_step)
            updates.append(target_update)

            np.savez(
                eval_module.log_path,
                times=times,
                updates=updates,
                timesteps=timesteps,
                returns=eval_module.evaluations_returns,
                successes=eval_module.evaluations_successes,
                sa_counts=all_sa_counts,
                # pi=all_pi,
                se=all_se,
                grad_accuracy=all_grad_accuracy,
                **ppo_logs,
                **props_logs,
            )

        # if do_ppo_update:


    # current_time = time.time() - start_time
    # print(f'Time: {current_time}')

    envs.close()


if __name__ == "__main__":
    main()
