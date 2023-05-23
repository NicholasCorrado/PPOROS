import os

from PPOROS.hyperparams.ppo_ros import PPOROS_PARAMS
from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def write_args():


    env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4', ]

    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/ros_b{buffer_size}.txt", "w")

            update_epochs = 10
            ros_update_epochs = 16

            target_kl = 0.03
            ros_clip_coef = 0.3
            ros_anneal_lr = 0

            params = PPOROS_PARAMS[env_id][buffer_size]
            if 'path' in params.keys():
                del params['path']

            args = gen_args(
                device='cpu',
                length="short",
                arg_generator=ppo_ros,
                env_id=env_id,
                buffer_size=buffer_size,
                update_epochs=update_epochs,
                target_kl=target_kl,
                ros_update_epochs=ros_update_epochs,
                ros_clip_coef=ros_clip_coef,
                ros_anneal_lr=ros_anneal_lr,
                stats=0,
                **PPOROS_PARAMS[env_id][buffer_size]
            )
            write_to_file(f, args)


if __name__ == "__main__":

    write_args()


