from PPOROS.scripts.best_hyperparams import PPO_PARAMS_NO_ANNEAL
from PPOROS.scripts.utils import ppo, gen_args, write_to_file, TIMESTEPS

def write_args():


    # env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Ant-v4', 'Walker2d-v4', 'InvertedPendulum-v4', 'InvertedDoublePendulum-v4', ]
    # env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Ant-v4', 'Walker2d-v4', 'InvertedDoublePendulum-v4', ]

    env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4', ]

    # env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']


    # env_ids = ['Swimmer-v4', 'Humanoid-v4']

    f = open(f"args/discrete.txt", "w")


    for env_id in env_ids:
        epochs = 10
        args = gen_args(
            device='cpu',
            length="short",
            arg_generator=ppo,
            env_id=env_id,
            total_timesteps=TIMESTEPS[env_id],
            update_epochs=epochs,
            stats=0,
            **PPO_PARAMS_NO_ANNEAL[env_id]
        )
        write_to_file(f, args)

if __name__ == "__main__":

    write_args()


