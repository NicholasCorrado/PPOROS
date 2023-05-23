import os

from PPOROS.hyperparams.ppo import PPO_PARAMS
from PPOROS.scripts.utils import ppo, gen_args, write_to_file, TIMESTEPS

def write_args():


    env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4', ]

    f = open(f"commands/ppo.txt", "w")

    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        epochs = 10
        target_kl = 0.03
        args = gen_args(
            device='cpu',
            length="short",
            arg_generator=ppo,
            env_id=env_id,
            target_kl=target_kl,
            total_timesteps=TIMESTEPS[env_id],
            update_epochs=epochs,
            stats=0,
            save_policy=True,
            **PPO_PARAMS[env_id]
        )
        write_to_file(f, args)

if __name__ == "__main__":

    write_args()


