PPOBUFFER_PARAMS = {
    'Swimmer-v4': {
        2: {
            'num_steps': 4096,
            'lr': 1e-3,
            'target_kl': 0.03
        },
        4: {
            'num_steps': 4096,
            'lr': 1e-3,
            'target_kl': 0.03,
        },
        8: {
            # 'num_steps': 1024,
            # 'lr': 1e-3,
            # 'target_kl': 0.03,
        },
    },
    'HalfCheetah-v4': {
        2: {
            'num_steps': 1024,
            'lr': 1e-4,
            'target_kl': 0.03,
        },
        4: {
            'num_steps': 1024,
            'lr': 1e-4,
            'target_kl': 0.03,
        },
        8: {
            # 'num_steps': 2048,
            # 'lr': 1e-4,
            # 'target_kl': 0.03,
        },
    },
    'Hopper-v4': {
        2: {
            'num_steps': 4096,
            'lr': 1e-4,
            'target_kl': 0.03,
        },
        4: {
            'num_steps': 1024,
            'lr': 1e-4,
            'target_kl': 0.03,
        },
        8: {
            # 'num_steps': 1024,
            # 'lr': 1e-3,
            # 'target_kl': 0.03,
        },
    },
    'Walker2d-v4': {
        2: {
            'num_steps': 4096,
            'lr': 1e-3,
            'target_kl': 0.03,
        },
        4: {
            'num_steps': 2048,
            'lr': 1e-3,
            'target_kl': 0.03,
        },
        8: {
            # 'num_steps': 8192,
            # 'lr': 1e-4,
            # 'target_kl': 0.03,
        },
    },
    'Ant-v4': {
        2: {
            'num_steps': 4096,
            'lr': 1e-4,
            'target_kl': 0.03,
        },
        4: {
            'num_steps': 8192,
            'lr': 1e-4,
            'target_kl': 0.03,
        },
        8: {
            # 'num_steps': 8192,
            # 'lr': 1e-4,
        },
    },
    'Humanoid-v4': {
        2: {
            'num_steps': 8192,
            'lr': 1e-4,
            'target_kl': 0.03,
        },
        4: {
            'num_steps': 4096,
            'lr': 1e-4,
            'target_kl': 0.03,
        },
        8: {
            # 'num_steps': None,
            # 'lr': None,
        },
    },
}