import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

import custom_envs





def value_iteration(env, max_iterations=100, theta=0.000001):
    shape = env.shape
    rows, cols = shape[0], shape[1]
    n_states = rows*cols
    v = np.zeros(n_states)
    q = np.zeros((n_states, 4))
    gamma = 0.99

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

                next_row = np.clip(next_row, 0, rows-1)
                next_col = np.clip(next_col, 0, cols-1)

                P[row, col, a, next_row, next_col] = 1
    P = P.reshape((n_states, 4, n_states))
    P[0, :, :] = 0
    # P[0, :, 0] = 1
    P[n_states-1, :, :] = 0
    # P[24, :, 24] = 1

    pi = np.ones((n_states, 4)) * 0.25

    r = env.rewards
    r = r.reshape(-1)

    # r = np.ones((rows, cols)) * -0.01
    # r[0, 0] = 0.5
    # r[4, 4] = 1
    # r = r.reshape(-1)

    for i in range(max_iterations):
        for s in range(1, n_states-1):
            for a in range(4):
                new_q = 0
                for ns in range(n_states):
                    new_q += P[s, a, ns] * (r[ns] + gamma*v[ns])
                q[s, a] = new_q
            v[s] = q[s, :] @ pi[s, :]

    A = q-np.tile(v, (4, 1)).T
    return A, q, v

if __name__ == '__main__':

    env = gym.make('GridWorld-10x10-v0')
    A, q, v = value_iteration(env, 100)
    # print(q)
    v = v.reshape((10,10))
    print(v.reshape(10,10))
    # print()

    # x = np.linspace(0, 5 - 1, 5) + 0.5
    # y = np.linspace(5 - 1, 0, 5) + 0.5
    # X, Y = np.meshgrid(x, y)
    # zeros = np.zeros((5, 5))
    #
    # fig = plt.figure(figsize=(10,10))
    # ax = plt.axes()
    # # Get max values
    # q_max = q.max(axis=1).reshape(5, 5)
    # q = q.reshape((5,5,4))
    # for i in range(5):
    #     for j in range(5):
    #         q_star = np.zeros((5, 5))
    #         q_max_s = q_max[i, j]
    #         max_vals = np.where(q_max_s == q[i, j])[0]
    #         for action in max_vals:
    #             q_star[i, j] = 0.4
    #             # Plot results
    #             if action == 0:
    #                 # Move up
    #                 plt.quiver(X, Y, zeros, q_star, scale=1, units='xy')
    #             elif action == 2:
    #                 # Move left
    #                 plt.quiver(X, Y, -q_star, zeros, scale=1, units='xy')
    #             elif action == 1:
    #                 # Move down
    #                 plt.quiver(X, Y, zeros, -q_star, scale=1, units='xy')
    #             elif action == 3:
    #                 # Move right
    #                 plt.quiver(X, Y, q_star, zeros, scale=1, units='xy')
    #
    # plt.xlim([0, 5])
    # plt.ylim([0, 5])
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # # plt.title(title)
    # plt.grid()
    # plt.show()