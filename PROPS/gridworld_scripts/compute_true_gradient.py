import os

import gymnasium as gym
import torch

import custom_envs
import numpy as np
from matplotlib import pyplot as plt

from PROPS.utils import AgentDiscrete, make_env


def compute_gradient(env, pi, s, a, A):
    # sa = np.zeros((25, 4))
    s = np.argmax(s, axis=-1).reshape(-1)
    a = a.reshape(-1)
    # grad = (A[s, a]*(1 - pi[s, a])).mean(axis=-1)
    grad = np.zeros((np.prod(env.shape), 4))
    for si, ai in zip(s, a):
        grad[si, ai] += A[si, ai] * (1 - pi[si, ai])

    grad = grad / len(s)

    return grad.reshape(-1)


def value_iteration(env, max_iterations=100, theta=0.000001):
    shape = env.shape
    rows, cols = shape[0], shape[1]
    n_states = rows * cols
    gamma = 0.99

    v = np.zeros(n_states)
    q = np.zeros((n_states, 4))
    r = env.rewards.reshape(-1)
    pi = np.ones((n_states, 4)) * 0.25

    #######################################
    ### START: CREATE TRANSITION MATRIX ###
    #######################################
    P = np.zeros((rows, cols, 4, rows, cols))
    for row in range(rows):
        for col in range(cols):
            for a in range(4):
                next_row = row
                next_col = col
                # up
                if a == 0:
                    next_row -= 1
                # down
                elif a == 1:
                    next_row += 1
                # left
                elif a == 2:
                    next_col -= 1
                # down
                elif a == 3:
                    next_col += 1

                next_row = np.clip(next_row, 0, rows - 1)
                next_col = np.clip(next_col, 0, cols - 1)

                P[row, col, a, next_row, next_col] = 1

    P = P.reshape((n_states, 4, n_states))
    P[0, :, :] = 0
    P[n_states - 1, :, :] = 0
    #####################################
    ### END: CREATE TRANSITION MATRIX ###
    #####################################

    # value iteration
    for i in range(max_iterations):
        for s in range(1, n_states - 1):
            for a in range(4):
                new_q = 0
                for ns in range(n_states):
                    new_q += P[s, a, ns] * (r[ns] + gamma * v[ns])
                q[s, a] = new_q
            v[s] = q[s, :] @ pi[s, :]

    A = q - np.tile(v, (4, 1)).T
    return A, q, v

def simulate(env, num_episodes):

    sa_counts = np.zeros(shape=(env.observation_space.shape[-1], env.action_space.n))
    all_obs = []
    all_actions = []

    for episode_i in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                # actions = agent.get_action(torch.Tensor(obs).to(device), noise=False)
                # actions = actions.cpu().numpy()
                actions = env.action_space.sample()

            s_idx = np.argmax(obs == 1)
            sa_counts[s_idx, actions] += 1

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            all_obs.append(obs)
            all_actions.append(actions)

            obs = next_obs

    return np.array(all_obs), np.array(all_actions), sa_counts

if __name__ == '__main__':
    env = gym.make('GridWorld-5x5-v0')
    obs, actions, sa = simulate(env, num_episodes=100000)

    sa_occupancy = sa / sa.sum()
    adv, q, v = value_iteration(env, 100)
    pi = np.ones(shape=(25, 4))*0.25
    grad = compute_gradient(env, pi, obs, actions, adv)

    print(len(obs))

    os.makedirs('data', exist_ok=True)
    np.save('data/grad_true.npy', grad)
    np.save('data/adv_true.npy', adv)
    np.save('data/q_true.npy', q)
    np.save('data/v_true.npy', v)
    np.save('data/sa_occupancy_true.npy', sa_occupancy)

