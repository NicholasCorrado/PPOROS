import argparse
import copy
import os
import pickle
import random
import time
from collections import defaultdict, deque
from distutils.util import strtobool

import gymnasium as gym
from torch.distributions import Categorical

import custom_envs
# import minatar
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from PROPS.ppo_props_discrete import update_ppo, update_props
from PROPS.utils import layer_init, EvaluateDiscrete, get_latest_run_id


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
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of episodes over which policies are evaluated")
    parser.add_argument("--results-dir", "-f", type=str, default="results",
                        help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")
    parser.add_argument("--results-subdir", "-s", type=str, default="",
                        help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")

    # General training parameters (both PROPS and PPO)
    parser.add_argument("--env-id", type=str, default="TwoStep-v0", help="Environment id")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=40000, help="Number of timesteps to train")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="If toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="If toggled, cuda will be enabled by default")

    # PPO hyperparameters
    parser.add_argument("--num-steps", type=int, default=200,
                        help="PPO target batch size (n in paper), the number of steps to collect between each PPO policy update")
    parser.add_argument("--buffer-batches", "-b", type=int, default=1,
                        help="Number of PPO target batches to store in the replay buffer (b in paper)")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.5, help="PPO Adam optimizer learning rate")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for PPO policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="General advantage estimation lambda (not the lambda used for PROPS")
    parser.add_argument("--num-minibatches", type=int, default=1, help="Number of minibatches updates for PPO update")
    parser.add_argument("--update-epochs", type=int, default=1, help="Number of epochs for PPO update")
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
    parser.add_argument("--props", type=int, default=0,
                        help="If True, use PROPS to collect data, otherwise use on-policy sampling")
    parser.add_argument("--ros", type=int, default=0,
                        help="If True, use ROS to collect data, otherwise use on-policy sampling")
    parser.add_argument("--props-num-steps", type=int, default=5,
                        help="PROPS behavior batch size (m in paper), the number of steps to run in each environment per policy rollout")
    parser.add_argument("--props-learning-rate", "-props-lr", type=float, default=1e-3,
                        help="PROPS Adam optimizer learning rate")
    parser.add_argument("--props-anneal-lr", type=lambda x: bool(strtobool(x)), default=0, nargs="?", const=False,
                        help="Toggle learning rate annealing for PROPS policy")
    parser.add_argument("--props-clip-coef", type=float, default=0.3,
                        help="Surrogate clipping coefficient \epsilon_PROPS for PROPS")
    parser.add_argument("--props-max-grad-norm", type=float, default=0.5,
                        help="Maximum norm for gradient clipping for PROPS update")
    parser.add_argument("--props-num-minibatches", type=int, default=1,
                        help="Number of minibatches updates for PROPS update")
    parser.add_argument("--props-update-epochs", type=int, default=16, help="Number of epochs for PROPS update")
    parser.add_argument("--props-target-kl", type=float, default=1,
                        help="Target/cutoff KL divergence threshold for PROPS update")
    parser.add_argument("--props-lambda", type=float, default=0, help="Regularization coefficient for PROPS update")
    parser.add_argument("--props-adv", type=int, default=False, help="If True, the PROPS update is weighted using the absolute advantage |A(s,a)|")
    parser.add_argument("--props-eval", type=int, default=False,
                        help="If set, the PROPS policy is evaluated every props_eval ")

    # Sampling error (se)
    parser.add_argument("--se", type=int, default=1,
                        help="If True, sampling error is computed every se_freq PPO updates.")
    parser.add_argument("--se-ref", type=int, default=0,
                        help="If True, on-policy sampling error is computed using the PPO policy sequence obtained while using PROPS. Only applies if se is True.")
    parser.add_argument("--se-lr", type=float, default=1e-3,
                        help="Adam optimizer learning rate used to compute the empirical (maximum likelihood) policy in sampling error computation.")
    parser.add_argument("--se-epochs", type=int, default=250,
                        help="Number of epochs to compute empirical (maximum likelihood) policy.")
    parser.add_argument("--se-freq", type=int, default=1, help="Compute sampling error very se_freq PPO updates")

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


class AgentDiscrete(nn.Module):
    def __init__(self, envs, relu=False):
        super().__init__()
        if isinstance(envs.single_observation_space, gym.spaces.Box):
            input_dim = np.array(envs.single_observation_space.shape).prod()
        else:
            input_dim = np.array(envs.single_observation_space.n)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(7, 1), std=1.0),
        )

        W = np.array([
            [0, 0],
            [0, 3.9],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ], dtype=np.float32)

        W = torch.from_numpy(W.T)
        W.requires_grad = True
        actor_layer = nn.Linear(input_dim, envs.single_action_space.n, bias=False)
        actor_layer.weight = torch.nn.Parameter(W)
        self.actor = nn.Sequential(
           actor_layer,
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.log_prob(action), probs.entropy(), self.critic(x),

    def get_action(self, x, noise=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if noise:
            action = probs.sample()
        else:
            action = probs.sample() # @TODO
        return action

    def get_action_and_info(self, x, action=None, clamp=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.log_prob(action), probs.entropy()

    def sample_actions(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action


    def get_traj_probs(self):
        s0 = torch.Tensor([[1, 0, 0, 0, 0, 0, 0]])
        sL = torch.Tensor([[0, 1, 0, 0, 0, 0, 0]])
        logits0 = self.actor(s0)
        logitsL = self.actor(sL)

        probs0 = Categorical(logits=logits0).probs.squeeze()
        probsL = Categorical(logits=logitsL).probs.squeeze()
        prob_opt = probs0[0] * probsL[0]
        prob_worst = probs0[0] * probsL[1]
        prob_subopt = probs0[1]

        return prob_opt, prob_worst, prob_subopt

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        return env

    return thunk

def compute_se(args, agent, rewards, sampling_error_logs):
    # COMPUTE SAMPLING ERROR

    prob_opt_expected, prob_worst_expected, prob_subopt_expected = agent.get_traj_probs()
    prob_opt_empirical, prob_worst_empirical, prob_subopt_empirical = (rewards == 2).sum()/100, (rewards == 0.5).sum()/100, (rewards == 1).sum()/100

    probs_expected = np.array([
        prob_opt_expected.item(),
        prob_worst_expected.item(),
        prob_subopt_expected.item(),
    ]) * 100
    probs_empirical = np.array([
        prob_opt_empirical.item(),
        prob_worst_empirical.item(),
        prob_subopt_empirical.item(),
    ]) * 100

    sampling_error_logs['traj_probs_expected'].append(probs_expected)
    sampling_error_logs['traj_probs_empirical'].append(probs_empirical)
    sampling_error_logs['traj_probs_error'].append(probs_empirical-probs_expected)

    # print(sampling_error_logs['traj_probs_expected'])
    np.savez(f'{args.save_dir}/stats.npz', **sampling_error_logs)

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
                compute_se(args, agent, rewards_buffer, sampling_error_logs)
                if args.se_ref:
                    compute_se_ref(args, agent_buffer, envs, next_obs_buffer, sampling_error_logs, global_step)
                    sampling_error_logs[f'diff_kl_mle_target'].append(
                        sampling_error_logs[f'kl_mle_target'][-1] - sampling_error_logs[f'ref_kl_mle_target'][-1])
                    print('(PROPS - On-policy) sampling error:', sampling_error_logs[f'diff_kl_mle_target'])
                    print('On-policy sampling error:', sampling_error_logs[f'ref_kl_mle_target'])

                sampling_error_logs['t'].append(global_step)
                print('PROPS sampling error:', sampling_error_logs[f'traj_probs_error'])
                np.savez(f'{args.save_dir}/stats.npz',
                         **sampling_error_logs)

                if args.track:
                    writer.add_scalar("charts/diff_kl_mle_target", sampling_error_logs[f'diff_kl_mle_target'][-1],
                                      global_step)

        # best_arm_count = (actions_buffer.detach().numpy() == 999).sum()
        # print('best arm count:', best_arm_count, )
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
        if args.props and global_step % args.props_num_steps == 0:  # and global_step > 25000:
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