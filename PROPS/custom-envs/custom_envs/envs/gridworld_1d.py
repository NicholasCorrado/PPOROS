from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class GridWorld1DEnv(gym.Env):
    def __init__(self, n=10, rewards=(-0.01, 0.5, 1)):
        super().__init__()

        self.n = n
        self.shape = (1, n)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(np.product(self.n),))

        self.pos = n//2
        self.init_pos = n//2  # agent starts in middle of the grid.

        self.rewards = rewards[0] * np.ones(self.n)
        self.rewards[0] = rewards[1] # subopt
        self.rewards[-1] = rewards[2] # opt

        self.terminals = np.zeros(self.n, dtype=bool)
        self.terminals[self.rewards > 0] = True

        print(self.rewards)
        print(self.terminals)


    def step(self, a):
        # up
        if a == 0:
            self.pos -= 1
        # down
        elif a == 1:
            self.pos += 1


        self.pos = np.clip(self.pos, a_min=0, a_max=self.n-1).astype(int)

        state = np.zeros(self.n)
        state[self.pos] = 1
        reward = self.rewards[self.pos]
        terminated = self.terminals[self.pos]
        truncated = False
        info = {}

        return state, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.pos = self.init_pos
        state = np.zeros(self.n)
        state[self.pos] = 1

        return state, {}