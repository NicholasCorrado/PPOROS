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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from PROPS.gridworld_scripts.compute_true_gradient import simulate, compute_gradient, value_iteration
from PROPS.utils import Evaluate, AgentDiscrete, EvaluateDiscrete, ConfigLoader, StoreDict, layer_init
from PROPS.utils import get_latest_run_id, make_env, Agent

class AgentDiscrete(nn.Module):
    def __init__(self, envs, linear):
        super().__init__()
        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 1), std=1.0),
        # )
        # self.actor = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        # )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), envs.single_action_space.n), std=0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64), std=0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def make_env(env_id, env_kwargs, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, **env_kwargs)
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
    parser.add_argument("--eval", type=int, default=1, help="Whether or not to evaluate target policy")
    parser.add_argument("--eval-freq", type=int, default=100*1, help="Evaluate PPO and/or PROPS policy every eval_freq PPO updates")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes over which policies are evaluated")
    parser.add_argument("--results-dir", "-f", type=str, default="results", help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")
    parser.add_argument("--results-subdir", "-s", type=str, default="", help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")
    parser.add_argument("--run-id", type=int, default=None, help="Results will be saved to <results_dir>/<env_id>/<subdir>/<algo>/run_<run_id>")

    # General training parameters (both PROPS and PPO)
    parser.add_argument("--env-id", type=str, default="GridWorld1D-10-v0", help="Environment id")
    parser.add_argument("--env-kwargs", type=str, nargs="*", action=StoreDict, default={}, help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=250000*1, help="Number of timesteps to train")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="If toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="If toggled, cuda will be enabled by default")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    # PPO hyperparameters
    parser.add_argument("--num-steps", type=int, default=100, help="PPO target batch size (n in paper), the number of steps to collect between each PPO policy update")
    # parser.add_argument("--num-traj", type=int, default=10*1, help="PPO target batch size, the number of trajectories to collect between each PPO policy update")
    parser.add_argument("--buffer-batches", "-b", type=int, default=1, help="Number of PPO target batches to store in the replay buffer (b in paper)")
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
    parser.add_argument("--linear", type=int, default=1, help="")
    parser.add_argument("--actor-critic", type=int, default=0, help="")
    parser.add_argument("--reinforce", type=int, default=0, help="")
    parser.add_argument("--oracle-adaptive", type=int, default=0, help="")
    parser.add_argument("--random-sampling", type=int, default=0, help="")
    parser.add_argument("--exact", type=int, default=0, help="")


    # PROPS/ROS hyperparameters
    parser.add_argument("--props", type=int, default=0, help="If True, use PROPS to collect data, otherwise use on-policy sampling")
    parser.add_argument("--ros", type=int, default=0, help="If True, use ROS to collect data, otherwise use on-policy sampling")
    parser.add_argument("--props-num-steps", type=int, default=10, help="PROPS behavior batch size (m in paper), the number of steps to run in each environment per policy rollout")
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
    parser.add_argument("--se-init", type=int, default=1, help="If True, sampling error is computed every se_freq PPO updates.")
    parser.add_argument("--se-ref", type=int, default=1, help="If True, on-policy sampling error is computed using the PPO policy sequence obtained while using PROPS. Only applies if se is True.")
    parser.add_argument("--se-lr", type=float, default=1e-3, help="Adam optimizer learning rate used to compute the empirical (maximum likelihood) policy in sampling error computation.")
    parser.add_argument("--se-epochs", type=int, default=250, help="Number of epochs to compute empirical (maximum likelihood) policy.")
    parser.add_argument("--se-freq", type=int, default=None, help="Compute sampling error very se_freq PPO updates")
    parser.add_argument("--se-debug", type=int, default=None, help="Only run PROPS when we evaluate sampling error")

    # loading pretrained models
    parser.add_argument("--policy-path", type=str, default=None, help="Path of pretrained policy to load")
    parser.add_argument("--normalization-dir", type=str, default=None, help="Directory contatining normalization statistics of pretrained policy")

    args = parser.parse_args()
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

    if args.reinforce:
        algo = 'reinforce'
        args.update_epochs = 1
        args.minibatch_size = args.buffer_size
        # args.ent_coef = 0
        args.gae_lambda = 1 # advantage estimates reduce to MC discounted return estimates
    elif args.actor_critic:
        algo = 'actor_critic'
        args.update_epochs = 1
        args.minibatch_size = args.buffer_size
        # args.ent_coef = 0
    else:
        algo = 'ppo'
    # algo = 'reinforce'
    # args.update_epochs = 1
    # args.ent_coef = 0
    # args.gae_lambda = 1 # advantage estimates reduce to MC discounted return estimates
    if args.props:
        sampling = 'props'
    elif args.ros:
        sampling = 'ros'
        args.props_num_steps = 1
        args.props_update_epochs = 1
        args.props_clip_coef = 9999999
        args.props_target_kl = 9999999
        args.props_lambda = 0
    else:
        sampling = 'on_policy'
        # args.props_num_steps = args.num_steps # to force num_props_updates = num_updates

    if args.oracle_adaptive:
        args.algo = 'oracle_adaptive'
    else:
        args.algo = f'{algo}_{sampling}'

    args.save_dir = f"{args.results_dir}/{args.env_id}/{args.algo}/{args.results_subdir}"

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
    # print(b_advantages)
    b_returns = returns.view(-1)
    b_values = values.view(-1)

    _, _, newlogprobs, entropy, newvalues = agent.get_action_and_value(b_obs, b_actions)
    newvalues = newvalues.view(-1)

    # pg_loss = -(b_returns.mean() * newlogprobs).mean()
    pg_loss = -(b_advantages * newlogprobs).mean()

    v_loss = 0.5 * ((newvalues - b_returns) ** 2).mean()
    entropy_loss = entropy.mean()

    loss = pg_loss + v_loss * args.vf_coef - args.ent_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()

    # grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    grad = [p.grad.reshape(-1) for p in agent.actor.parameters() if p.grad is not None and p.requires_grad]
    grad = torch.concat([grad[0]])

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




def update_props(agent_props, envs, props_optimizer, obs, logprobs, actions, advantages, global_step, args, writer):
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
    b_obs = obs[start:end].reshape((-1,) + envs.single_observation_space.shape).to(args.device)
    b_actions = actions[start:end].reshape((-1,) + envs.single_action_space.shape).to(args.device)
    # b_logits = logits[start:end].reshape(-1)  # action logits for PPO policy
    with torch.no_grad():
        _, _, logprobs, _, _ = agent_props.get_action_and_value(b_obs, b_actions)
    b_logprobs = logprobs.reshape(-1).to(args.device)

    b_probs = torch.exp(logprobs).to(args.device)

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


                approx_kl_to_log = approx_kl

            kl_regularizer_loss = (mb_probs*(mb_logprobs - props_logprobs)).mean()

            pg_loss1 = props_ratio
            pg_loss2 = torch.clamp(props_ratio, 1 - args.props_clip_coef, 1 + args.props_clip_coef)
            if args.props_adv:
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_abs_advantages).mean()
            else:
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            if args.ros:
                pg_loss = props_logratio.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss + args.props_lambda * kl_regularizer_loss

            props_optimizer.zero_grad()
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(agent_props.parameters(), args.props_max_grad_norm)
            grad_norms.append(grad_norm.detach().cpu().numpy())

            props_optimizer.step()
            num_update_minibatches += 1

        if args.props_target_kl:
            # print(approx_kl)
            if approx_kl > args.props_target_kl:
                done_updating = True
                break

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
            'props_kl_regularizer_loss': float(kl_regularizer_loss.item()),
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
            writer.add_scalar("props/kl_regularizer_loss", kl_regularizer_loss.item(), global_step)
            writer.add_scalar("props/approx_kl", approx_kl_to_log, global_step)
    return props_stats

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
        [make_env(args.env_id, args.env_kwargs, args.seed + i, i, capture_video, run_name) for i in range(args.num_envs)]
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
    eval_module = EvaluateDiscrete(
        model=agent,
        eval_env=None,
        n_eval_episodes=args.eval_episodes,
        log_path=args.save_dir,
        device=args.device)
    eval_module_props = EvaluateDiscrete(
        model=agent_props,
        eval_env=None,
        n_eval_episodes=args.eval_episodes,
        log_path=args.save_dir,
        device=args.device,
        suffix='props')

    args.buffer_size = args.num_steps * args.buffer_batches
    obs_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    actions_buffer = torch.zeros((args.buffer_size, args.num_envs) + envs.single_action_space.shape).to(args.device)
    rewards_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(args.device)
    dones_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(args.device)
    values_buffer = torch.zeros((args.buffer_size, args.num_envs)).to(args.device)
    buffer_pos = 0  # index of buffer position to be updated in the current timestep

    args.props_minibatch_size = int((args.buffer_size - args.props_num_steps) // args.props_num_minibatches)

    # initialize RL loop
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    next_terminated = torch.zeros(args.num_envs).to(args.device)

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
    all_grad_empirical = []
    all_grad_true = []
    all_sa_occupancy = []
    all_sa_occupancy_true = []

    start_time = time.time()
    global_step = 0
    target_update = 0
    props_update = 0

    ####################################
    ### START: COMPUTE TRUE GRADIENT ###
    ####################################

    # grad_true = np.load('gridworld_scripts/data/grad_true.npy')
    # adv_true = np.load('gridworld_scripts/data/adv_true.npy')

    # env = gym.make(args.env_id, **args.env_kwargs)
    # obs, actions, sa = simulate(env, num_episodes=10000)

    # sa_occupancy_true = sa / sa.sum()
    # adv_true, q_true, v_true = value_iteration(env, 100)
    # pi = np.ones(shape=(25, 4))*0.25
    # grad_true = compute_gradient(env, pi, obs, actions, adv_true)
    sa_occupancy_true = np.load(f'../gridworld_clean/data/{args.env_id}/sa_occupancy_true.npy')
    grad_true = np.load(f'../gridworld_clean/data/{args.env_id}/grad_true.npy')
    adv_true = np.load(f'../gridworld_clean/data/{args.env_id}/adv_true.npy')

    grad_true_norm = np.linalg.norm(grad_true)

    ##################################
    ### END: COMPUTE TRUE GRADIENT ###
    ##################################

    if args.props_eval:
        eval_module_props.evaluate(global_step, train_env=envs, noise=False)

    ppo_stats = {}
    props_stats = {}

    all_se = []
    all_sa_counts = []
    sa_counts = np.zeros(shape=(envs.single_observation_space.shape[-1], envs.single_action_space.n))
    possible_actions = np.arange(envs.single_action_space.n)


    theta = np.zeros((25, 4))
    for global_step in range(args.total_timesteps):
        # collect a transition
        global_step += 1 * args.num_envs
        obs_buffer[buffer_pos] = next_obs # store unnormalized obs
        dones_buffer[buffer_pos] = next_done

        with torch.no_grad():
            if args.exact:
                s_idx = np.argmax(next_obs)
                theta_s = theta[s_idx]
                pi = np.exp(theta_s)/np.sum(np.exp(theta_s))
                a_idx = np.random.choice(possible_actions, p=pi)
                action = torch.Tensor([a_idx])
                values = torch.Tensor([0])

            elif args.oracle_adaptive:
                s_idx = np.argmax(next_obs)
                sa = sa_counts[s_idx]
                pi = agent.get_pi_at_s(next_obs)[0]

                if np.sum(sa) == 0:
                    a_idx = np.random.choice(possible_actions, p=pi)
                else:
                    pi_empirical = sa / np.sum(sa)
                    a_idx = np.argmin(pi_empirical - agent.get_pi_at_s(next_obs))

                action = torch.Tensor([a_idx])

                _, _, logprobs, _, values = agent_props.get_action_and_value(next_obs, action)
            elif args.props or args.ros:
                action, logprobs, entropy, values = agent_props.get_action_and_value(next_obs)
                # fetch value and logprob w.r.t target policy (not behavior policy)
                _, _, logprobs, _, values = agent_props.get_action_and_value(next_obs, action)

                a_idx = action[0]
            else:
                action, logprobs, entropy, values = agent.get_action_and_value(next_obs)
                a_idx = action[0]

            print(values)
            actions_buffer[buffer_pos] = action
            values_buffer[buffer_pos] = values

        s_idx = np.where(next_obs[0] == 1)[0][0]
        sa_counts[s_idx, a_idx] += 1

        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        next_obs, next_done, next_terminated = torch.Tensor(next_obs).to(args.device), torch.Tensor(terminated | truncated).to(args.device), torch.Tensor(next_terminated).to(args.device)

        rewards_buffer[buffer_pos] = torch.tensor(reward).to(args.device).view(-1)
        dones_buffer[buffer_pos] = next_done

        for info in infos.get("final_info", []):
            # Skip the envs that are not done
            if info is None:
                continue

            if args.track:
                writer.add_scalar("charts/props_train_ret", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episode_length", info["episode"]["l"], global_step)

        buffer_pos += 1
        buffer_pos %= args.buffer_size

        # if terminated: print(reward)

        # determine what all needs to be done at this timestep
        do_ppo_update = ((global_step+1) % args.num_steps == 0) or args.exact
        do_props_update = (args.props and (global_step+1) % args.props_num_steps == 0) or args.ros
        do_eval = (global_step + 1) % args.eval_freq == 0

        if do_ppo_update or do_props_update:
            if global_step < args.buffer_size:
                obs = obs_buffer
                actions = actions_buffer
                rewards = rewards_buffer
                dones = dones_buffer
                values = values_buffer
            else:
                # right shift buffers so that the data is ordered from oldest to youngest
                obs = torch.roll(obs_buffer, buffer_pos)
                actions = torch.roll(actions_buffer, buffer_pos)
                rewards = torch.roll(rewards_buffer, buffer_pos)
                dones = torch.roll(dones_buffer, buffer_pos)
                values = torch.roll(values_buffer, buffer_pos)

            # Store the b previous target policies. We do this so we can compute on-policy sampling error with respect to
            # the target policy sequence obtained by PROPS.
            # if global_step % args.num_steps == 0:
            #     next_obs_buffer.append(copy.deepcopy(next_obs))
            #     agent_buffer.append(copy.deepcopy(agent))

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(args.device)
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

            # PPO update
            if do_ppo_update:
                target_update += 1

                # flatten the batch
                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                args.batch_size = int(args.num_envs * args.num_steps)
                args.minibatch_size = int(args.batch_size // args.num_minibatches)

                # Optimizing the policy and value network
                b_inds = np.arange(args.batch_size)
                clipfracs = []
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, args.batch_size, args.minibatch_size):
                        end = start + args.minibatch_size
                        mb_inds = b_inds[start:end]

                        _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],
                                                                                      b_actions.long()[mb_inds])
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

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break


                #
                # if args.exact:
                #     eval_returns, eval_obs, eval_actions, eval_rewards, sa_eval = eval_module.simulate(train_env=envs)
                #     A_init, q_init, v_init = value_iteration(envs.envs[0].unwrapped, 20)
                #     grad_true = compute_gradient(envs.envs[0].unwrapped, agent.get_pi(), eval_obs, eval_actions, A_init)
                #     theta += args.learning_rate*grad_true.reshape(25, 4)
                # else:
                #     # ppo_stats = update_ppo(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer)
                #     ppo_stats, grad_empirical = update_reinforce(agent, optimizer, envs, obs, logprobs, actions, advantages, returns, values, args, global_step, writer)

            if do_props_update:
                props_update += 1
                # Set props policy equal to current target policy
                for source_param, dump_param in zip(agent_props.parameters(), agent.parameters()):
                    source_param.data.copy_(dump_param.data)
                props_stats = update_props(agent_props, envs, props_optimizer, obs, logprobs, actions, advantages, global_step, args, writer)

        # Evaluate agent performance
        if do_eval:
            current_time = time.time() - start_time
            print(f"Training time: {int(current_time)} \tstep: {global_step+1} \tsteps per sec: {int(global_step / current_time)}")
            agent = agent.to(args.device)
            agent_props = agent_props.to(args.device)
            # # Evaluate PPO policy
            if args.eval:
                if args.exact:
                    target_ret, target_std, sa_eval = eval_module.evaluate(global_step, train_env=envs, noise=False, theta=theta)
                else:
                    target_ret, target_std, sa_eval = eval_module.evaluate(global_step, train_env=envs, noise=False)


            if global_step < args.buffer_size:
                obs = obs_buffer
                actions = actions_buffer
            else:
                # right shift buffers so that the data is ordered from oldest to youngest
                obs = np.roll(obs_buffer, buffer_pos)
                actions = np.roll(actions_buffer, buffer_pos)

            grad_empirical = compute_gradient(envs.envs[0].unwrapped, agent.get_pi(), obs.detach().numpy(), actions.detach().numpy().astype(int), adv_true)
            # grad_empirical_norm = np.linalg.norm(grad_empirical)

            all_grad_empirical.append(grad_empirical)
            all_grad_true.append(grad_true)

            print(grad_empirical.shape)

            grad_accuracy = (grad_empirical @ grad_true)/np.linalg.norm(grad_empirical)/grad_true_norm
            all_grad_accuracy.append(grad_accuracy.item())

            sa_occupancy = sa_counts/sa_counts.sum()
            # se = (sa_occupancy - sa_occupancy_true).sum()
            se = np.abs(sa_occupancy - sa_occupancy_true).sum()
            all_se.append(se)
            all_sa_occupancy.append(sa_occupancy)
            all_sa_occupancy_true.append(sa_occupancy_true)

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
                # sa_counts=all_sa_counts,
                sa_occupancy=all_sa_occupancy,
                sa_occupancy_true=all_sa_occupancy_true,
                # pi=all_pi,
                se=all_se,
                grad_accuracy=all_grad_accuracy,
                grad=all_grad_empirical,
                grad_true=all_grad_true,
                **ppo_logs,
                **props_logs,
            )


    envs.close()


if __name__ == "__main__":
    main()
