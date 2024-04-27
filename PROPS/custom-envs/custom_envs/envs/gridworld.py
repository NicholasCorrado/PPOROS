from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box


class GridWorldEnv(gym.Env):
    Left = 0
    Right = 1
    Down = 2
    Up = 3
    MaxStep = 100
    gamma = 1

    action_space = Discrete(4)
    observation_space = Box(0, 1, (8*8,))

    def __init__(self, mean_factor=1., scale_noise=0.) -> None:
        super().__init__()
        self.ready = False
        self.time_step = 0
        self.state = np.zeros(2)

        self.terminal_state = np.array([7, 7])
        self.trap_state = np.array([5, 5])
        self.reward_state = np.array([1, 7])

        self.mean_factor = mean_factor
        self.scale_noise = scale_noise

    def step(self, action: int):
        assert self.ready, "please reset."
        assert action in (GridWorldEnv.Up, GridWorldEnv.Down, GridWorldEnv.Left, GridWorldEnv.Right)
        self.time_step += 1
        old_state: np.ndarray = self.state.copy()

        i = action // 2
        j = action % 2
        self.state[i] += 1 if j else -1
        self.state = self.state.clip(0, 7)

        terminated = False
        truncated = self.time_step == GridWorldEnv.MaxStep

        if (self.state == old_state).all():
            r = -1
        else:
            if (self.state == self.terminal_state).all():
                r, terminated = 1, True
            # elif (self.state == self.trap_state).all():
            #     r = -10
            # elif (self.state == self.reward_state).all():
            #     r = 1
            else:
                r = -0.1

        if terminated or truncated:
            self.ready = False

        r *= self.mean_factor
        if self.scale_noise:
            r += self.scale_noise * (2 * np.random.random() - 1)  # [-scale_noise, scale_noise]

        return self.to_observation(), r, terminated, truncated, {}

    def to_observation(self):
        # return self.state.copy()
        idx = self.state[0] * 8 + self.state[1]
        obs = np.zeros(64)
        obs[int(idx)] = 1
        return obs

    def to_state(self, state):
        # return state
        return np.array([state // 8, state % 8])

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.state = np.zeros(2)
        self.ready = True
        self.time_step = 0
        return self.to_observation(), {}

    def render(self, mode='human'):
        return
        image = np.zeros((4, 4))

        image[3 - self.terminal_state[1], self.terminal_state[0]] = "T"
        image[3 - self.trap_state[1], self.trap_state[0]] = "t"
        image[3 - self.reward_state[1], self.reward_state[0]] = "r"
        image[3 - self.state[1], self.state[0]] = "X"
        print(image)

    def seed(self, seed=None):
        np.random.seed(seed)


def play():
    env = GridWorldEnv()
    s, _ = env.reset()
    done = False
    while not done:
        env.render()
        a = env.action_space.sample()
        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        print(env.to_state(s), a, r)
        s = s_


if __name__ == '__main__':
    play()