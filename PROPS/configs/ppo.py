

PPO_PARAMS = {
    'Swimmer-v4':
        {
            'num_steps': 2048,
            'lr': 1e-3,
        },
    'HalfCheetah-v4':
        {
            'num_steps': 1024,
            'lr': 1e-4,
        },
    'Hopper-v4':
        {
            'num_steps': 1024,
            'lr': 1e-4,
        },
    'Walker2d-v4':
        {
            'num_steps': 8192,
            'lr': 1e-3,
        },
    'Ant-v4':
        {
            'num_steps': 1024,
            'lr': 1e-4,
        },
    'Humanoid-v4':
        {
            'num_steps': 8192,
            'lr': 1e-4,
        },
}


OLD_PPO_PARAMS_ANNEAL = {
    'Swimmer-v4':
        {
            'num_steps': 2048,
            'lr': 1e-3,
        },
    'HalfCheetah-v4':
        {
            'num_steps': 1024,
            'lr': 1e-4,
        },
    'Hopper-v4':
        {
            'num_steps': 2048,
            'lr': 5e-4,
        },
    'Walker2d-v4':
        {
            'num_steps': 8192,
            'lr': 5e-4,
        },
    'Ant-v4':
        {
            'num_steps': 1024,
            'lr': 1e-4,
        },
    'Humanoid-v4':
        {
            'num_steps': 8192,
            'lr': 1e-4,
        },
}
OLD_PPO_PARAMS_NO_ANNEAL = {
    'Swimmer-v4': # done
        {
            'num_steps': 2048,
            'lr': 1e-3,
            'target_kl': 0.05, # or 0.05
        },
    'HalfCheetah-v4': # done
        {
            'num_steps': 4096,
            'lr': 1e-4,
            'target_kl': 0.05 # or 0.05
        },
    'Hopper-v4': # done
        {
            'num_steps': 2048,
            'lr': 1e-4,
            'target_kl': 0.05, # or 0.05
        },
    'Walker2d-v4': # done
        {
            'num_steps': 4096,
            'lr': 1e-3,
            'target_kl': 0.05,
        },
    'Ant-v4':
        {
            'num_steps': 2048,
            'lr': 1e-4,
            'target_kl': 0.05, # or 0.05
        },
    'Humanoid-v4': # done
        {
            'num_steps': 8192,
            'lr': 1e-4,
            'target_kl': 0.05, # or 0.5
        },
}
