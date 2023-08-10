import os

from gymnasium.envs.registration import register

ENVS_DIR = os.path.join(os.path.dirname(__file__), 'envs')

############################################################################
### Toy

register(
    id="Goal2D-v0",
    entry_point="custom_envs.envs.goal2d:Goal2DEnv",
    max_episode_steps=100,
)

register(
    id="Bandit5-v0",
    entry_point="custom_envs.envs.bandit:BanditEnv",
    max_episode_steps=1,
    kwargs={
        'n': 5
    }
)


register(
    id="Bandit100-v0",
    entry_point="custom_envs.envs.bandit:BanditEnv",
    max_episode_steps=1,
    kwargs={
        'n': 100
    }
)

register(
    id="Bandit1000-v0",
    entry_point="custom_envs.envs.bandit:BanditEnv",
    max_episode_steps=1,
    kwargs={
        'n': 1000
    }
)

for n in [10, 50, 100]:
    register(
        id=f"Discrete2D{n}-v0",
        entry_point="custom_envs.envs.discrete2d:Discrete2DEnv",
        max_episode_steps=50,
        kwargs={
            'n': n
        }
    )