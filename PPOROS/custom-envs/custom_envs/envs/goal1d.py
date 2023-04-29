from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import seaborn
from matplotlib import pyplot as plt


class Goal1DEnv(gym.Env):
    def __init__(self, delta=0.01):

        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), shape=(1,))

        self.boundary_low = np.array([-1])
        self.boundary_high = np.array([+1])
        self.observation_space = gym.spaces.Box(self.boundary_low, self.boundary_high, shape=(1,), dtype="float64")

        self.delta = delta
        self.step_count = 0
        super().__init__()

    def _clip_position(self):
        # Note: clipping makes dynamics nonlinear
        self.x = np.clip(self.x, self.boundary_low, self.boundary_high)

    def step(self, a):
        self.step_count += 1
        # a[0] = 1
        self.x += a[0] * self.delta
        self._clip_position()

        terminated = False
        truncated = False

        if self.x[0] > 0:
            reward = self.x[0]*0.5
        else:
            reward = 1*(np.tanh((-10*(2*self.x[0]+0.5)))+1)/1

        # if self.step_count == 100:
        #     if self.x[0] > 0: print('right')
        #     else: print('left')

        info = {}
        return self.x, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.x = np.array([0.0])
        self.step_count = 0

        # self.x = np.random.uniform(low=[0, 0.1], high=[0.2, 0.1])

        return self.x, {}


if __name__ == "__main__":

    seaborn.set_theme()
    n = 100
    x = np.linspace(-1, 1, n)
    y = x**2

    xhalf = x[:n//2]
    y[:n//2] = 1*(np.tanh((-10*(2*x[:n//2]+0.5)))+1)

    # y = np.tanh((-10*(x)))

    plt.plot(x,y)
    plt.xlabel('x position')
    plt.ylabel('reward')
    plt.title('Reward Function')
    plt.show()