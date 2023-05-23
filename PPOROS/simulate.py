import argparse
import time

import gym
import gymnasium as gym
import numpy as np

# from augment.rl.algs.ddpg import DDPG
import torch
from ppo_ros_discrete import Agent


def simulate(env, model, num_episodes, seed=0, render=False, flatten=True, verbose=False):
    '''
    Simulate a policy model in environment env for num_episodes episodes.

    :param env: a gym environment
    :param model: a policy that implements a predict() function with the following signature:
        def predict(observation: np.ndarray, deterministic: bool):
            // return action based on observation.
            // next_hidden_state is only relevant for recurrent policies. You can next_hidden_state=None if you don't have a recurrent policy.
            return action, next_hidden_state
    :param num_episodes: number of episodes
    :param seed: seed
    :param render: renders environment if True
    :param flatten: If false, transition data is separated by episode,
    e.g. observations = [ [obs_11, obs_12, ... obs_1n], # obs for episode 1
                          [obs_21, obs_22, ... obs_2n]  # obs for episode 2
                          ...
                        ]
    If True, transition data across multiple episodes is flattened to a 2D array,
    e.g. observations = [ obs_1, obs_2, ..., obs_n]
    If your environment can terminate early, you must have flatten=True, since numpy doesn't support jagged arrays.
    :param verbose: if True, prints the return of each episode.
    :return: observations, next_observations, actions, rewards, dones, infos as arrays
    '''
    np.random.seed(seed)

    observations, next_observations, actions, rewards, dones, infos = [], [], [], [], [], []
    returns = []

    for i in range(num_episodes):
        ep_observations, ep_next_observations, ep_actions, ep_rewards, ep_dones, ep_infos,  = [], [], [], [], [], []
        obs, _ = env.reset()
        done = False
        step = 1
        while not done:

            # if model is None, sample actions uniformly at random
            if model:
                action = model.get_action(torch.from_numpy(obs)).numpy()
            else:
                action = env.action_space.sample()


            ep_actions.append(action)
            ep_observations.append(obs)

            obs, reward, termianted, truncated, info = env.step(action)
            done = termianted | truncated

            ep_next_observations.append(obs)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_infos.append(info)

            if render:
                env.render()

            step += 1

        returns.append(sum(ep_rewards))

        if flatten:
            observations.extend(ep_observations)
            next_observations.extend(ep_next_observations)
            actions.extend(ep_actions)
            rewards.extend(ep_rewards)
            dones.extend(ep_dones)
            infos.extend(ep_infos)
        else:
            observations.append(ep_observations)
            next_observations.append(ep_next_observations)
            actions.append(ep_actions)
            rewards.append(ep_rewards)
            dones.append(ep_dones)
            infos.append(ep_infos)
        if verbose:
            print(f'episode {i}: return={returns[-1]}',)

    print(f'Average return over {num_episodes} episodes: {np.average(returns)} +/- {np.std(returns)}')
    return np.array(observations), np.array(next_observations), np.array(actions), np.array(rewards), np.array(dones), np.array(infos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()


    # env_kwargs = {'render_mode': 'human'}
    env_kwargs = {'render_mode': 'human'}
    env = gym.make(args.env_id, **env_kwargs)
    model = torch.load(f'policies/{args.env_id}.zip')
    model = None

    observations, next_observations, actions, rewards, dones, infos = simulate(
        env=env, model=model, num_episodes=args.num_episodes, seed=args.seed, render=True, flatten=True, verbose=True)


    '''
episode 8: return=-50.0
[ 0.11855708 -0.27298401  0.98876152 -2.36816768]
[ 0.11867547 -0.27431996  0.98959232 -2.37912166]    '''