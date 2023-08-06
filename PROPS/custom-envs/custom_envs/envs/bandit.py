from typing import Optional

import gymnasium as gym
import numpy as np


class BanditEnv(gym.Env):
    def __init__(self, n=1000):

        self.n = n
        self.action_space = gym.spaces.Discrete(self.n)
        self.observation_space = gym.spaces.Box(low=0, high=1)
        super().__init__()

        self.means = np.random.uniform(0, 0.9, self.n)
        self.stds = np.random.uniform(0, 1, self.n)
        self.means[0] = 1

    def step(self, a):

        reward = np.random.normal(self.means[a], self.stds[a])
        terminated = True
        truncated = False
        info = {}
        return np.array([0]), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return np.array([0]), {}