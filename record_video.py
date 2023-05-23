import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    x = np.arange(500)
    y = np.cos(2*np.pi*2*x/500)
    plt.plot(x,y)
    plt.show()

    env = gym.make('CartPole-v1',render_mode='rgb_array')
    env = RecordVideo(env, video_folder='videos')

    s, info = env.reset()
    done = False
    for t in range(100):
        action = np.random.randint(2)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        if done: break

    env.close()

