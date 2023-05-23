import os

from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def write_args():


    env_ids = ['Humanoid-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/ros_b{buffer_size}.txt", "w")

            num_steps = 8192
            ros_num_steps = 8192

            lr = 1e-4
            ros_lr = 1e-4

            update_epochs = 10
            ros_update_epochs = 16

            target_kl = 0.03
            ros_target_kl = 0.05

            ros_lambda = 0.1
            ros_clip_coef = 0.3
            ros_anneal_lr = 0

            for ros_num_steps in [256, 512, 1024, 2048]:
                for ros_target_kl in [0.03, 0.05, 0.1]:
                    for ros_lambda in [0.01, 0.1, 0.3]:
                        for ros_mb in [32]:

                            good = False
                            if ros_lambda == 0.01:
                                if ros_num_steps in [512,1024]:
                                   good = True
                            if ros_lambda == 0.1:
                                if ros_num_steps in [256, 2048]:
                                   good = True
                                if ros_num_steps in [1024] and ros_target_kl == 0.03:
                                    good = True
                            if ros_lambda == 0.3:
                                if ros_num_steps in [256] and ros_target_kl == 0.03:
                                   good = True

                            if not good: continue
                            args = gen_args(
                                device='cpu',
                                length="short",
                                arg_generator=ppo_ros,
                                env_id=env_id,
                                num_steps=num_steps,
                                lr=lr,
                                update_epochs=update_epochs,
                                target_kl=target_kl,
                                ros_lr=ros_lr,
                                buffer_size=buffer_size,
                                ros_num_steps=ros_num_steps,
                                ros_update_epochs=ros_update_epochs,
                                ros_target_kl=ros_target_kl,
                                ros_clip_coef=ros_clip_coef,
                                ros_anneal_lr=ros_anneal_lr,
                                ros_mb=ros_mb,
                                stats=0,
                                ros_lambda=ros_lambda
                            )
                            write_to_file(f, args)


if __name__ == "__main__":

    write_args()


