from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class GridWorldEnv(gym.Env):
    def __init__(self, shape=(5,5)):
        super().__init__()

        self.shape = np.array(shape)
        self.action_space = gym.spaces.Discrete(4)
        print(self.shape)
        self.observation_space = gym.spaces.Box(low=0, high=0, shape=(np.product(self.shape),))
        # self.observation_space = gym.spaces.Discrete(np.product(self.shape))

        self.rowcol = np.zeros(2)
        self.goal_rowcol = self.shape-1

    def _rowcol_to_obs(self, rowcol, goal_rowcol):
        # idx = np.concatenate([
        #     rowcol[0] * self.shape[0] + rowcol[1],
        #     goal_rowcol[0] * self.shape[0] + goal_rowcol[1],
        # ])

        idx = int(rowcol[0] * self.shape[0] + rowcol[1])
        state = np.zeros(self.observation_space.shape[-1])
        state[idx] = 1
        return state


    def step(self, a):

        # up
        if a == 0:
            self.rowcol[0] -= 1
        # down
        elif a == 1:
            self.rowcol[0] += 1
        # left
        elif a == 2:
            self.rowcol[1] -= 1
        # down
        elif a == 3:
            self.rowcol[1] += 1

        self.rowcol = np.clip(self.rowcol, a_min=np.zeros(2), a_max=self.shape-1)

        terminated = False
        truncated = False

        if np.all(self.rowcol == self.goal_rowcol):
            reward = 1
            terminated = True
        else:
            reward = 0

        state = self._rowcol_to_obs(self.rowcol, self.goal_rowcol)
        info = {}
        return state, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.rowcol = np.zeros(2)
        self.goal_rowcol = np.array(self.shape)-1
        state = self._rowcol_to_obs(self.rowcol, self.goal_rowcol)

        return state, {}
    #
    # def seed(self, seed):
    #     return
