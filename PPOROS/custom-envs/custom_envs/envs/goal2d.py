from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class Goal2DEnv(gym.Env):
    def __init__(self, delta=0.05):

        self.action_space = gym.spaces.Box(low=np.array([0, -np.pi]), high=np.array([1, np.pi]), shape=(2,))

        self.boundary_low = np.array([-1, -1])
        self.boundary_high = np.array([1, 1])
        self.observation_space = gym.spaces.Box(self.boundary_low, self.boundary_high, shape=(2,), dtype="float64")

        self.delta = delta
        super().__init__()

    def _clip_position(self):
        # Note: clipping makes dynamics nonlinear
        self.x = np.clip(self.x, self.boundary_low, self.boundary_high)

    def step(self, a):

        ux = a[0]*np.cos(a[1])
        uy = a[0]*np.sin(a[1])
        u = np.array([ux, uy])

        self.x += u * self.delta
        self._clip_position()

        terminated = False
        truncated = False

        # # print(self.x)
        # reward = -(self.x[0] - 0.9)**2
        # # reward = 0
        # # print(self.x)
        # is_success = False
        # if self.x[0] > 0.9 and np.abs(self.x[1]) < 0.1:
        #     # print('here')
        #     reward += 10
        #     terminated = True
        #     is_success = True
        # elif np.abs(self.x[1]) > 0.1:
        #     reward += 0
        #     terminated = True
        # else:
        #     reward = -0.1
        # if terminated: print(reward)
        # info = {'is_success': is_success}
        dist = np.linalg.norm(self.x)
        dist = np.linalg.norm(self.x)
        if self.x[0] > 0 and self.x[1] > 0:
            reward = 1.05*dist
        else:
            reward = dist

        info = {}
        return self.x, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.x = np.array([0.0, 0.0])
        # self.x = np.random.uniform(low=[0, 0.1], high=[0.2, 0.1])

        return self.x, {}
