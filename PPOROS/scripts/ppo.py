from PPOROS.scripts.utils import ppo, gen_args, write_to_file, TIMESTEPS

BEST_PPO_PARAMS = {
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

def write_args():


    # env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Ant-v4', 'Walker2d-v4', 'InvertedPendulum-v4', 'InvertedDoublePendulum-v4', ]
    # env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Ant-v4', 'Walker2d-v4', 'InvertedDoublePendulum-v4', ]

    env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4', ]

    # env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']


    # env_ids = ['Swimmer-v4', 'Humanoid-v4']

    f = open(f"args/discrete.txt", "w")


    for env_id in env_ids:
        for lr in [1e-3, 1e-4]:

            num_steps_list = [1024, 2048, 4096]
            if env_id in ['Ant-v4']:
                num_steps_list = [1024, 2048, 4096, 8192]
            if env_id in ['Humanoid-v4']:
                num_steps_list = [4096, 8192]

            for num_steps in num_steps_list:
                for target_kl in [0.05, 0.1]:
                    epochs = 10
                    args = gen_args(
                        device='cpu',
                        length="short",
                        arg_generator=ppo,
                        env_id=env_id,
                        lr=lr,
                        target_kl=target_kl,
                        num_steps=num_steps,
                        total_timesteps=TIMESTEPS[env_id],
                        update_epochs=epochs,
                        stats=0,
                    )
                    write_to_file(f, args)

if __name__ == "__main__":

    write_args()


