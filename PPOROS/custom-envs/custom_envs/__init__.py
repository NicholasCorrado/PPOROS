from gymnasium.envs.registration import register

register(
    id="Goal2D-v0",
    entry_point="custom_envs.envs:Goal2DEnv",
    max_episode_steps=100,
)

register(
    id="Goal1D-v0",
    entry_point="custom_envs.envs:Goal1DEnv",
    max_episode_steps=100,
)