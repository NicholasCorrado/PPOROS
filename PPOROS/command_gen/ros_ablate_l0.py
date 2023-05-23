import os

from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def args_hopper():

    env_ids = ['Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'Ant-v4', 'Humanoid-v4', ]
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/ros_b{buffer_size}.txt", "w")
            if env_id in ['Swimmer-v4']:
                num_steps_list = [2048,4096]
            if env_id in ['Walker2d-v4']:
                num_steps_list = [1024, 2048]
            if env_id in ['Hopper-v4']:
                num_steps_list = [1024, 2048]
            if env_id in ['Walker2d-v4']:
                num_steps_list = [2048]
            if env_id in ['Ant-v4']:
                num_steps_list = [1024, 2048]
            if env_id in ['Humanoid-v4']:
                num_steps_list = [8192]

            for num_steps in num_steps_list:

                lrs = []
                if env_id in ['Walker2d-v4', 'Hopper-v4',  'Swimmer-v4']:
                    lrs = [1e-3]
                elif env_id in ['Ant-v4', 'Humanoid-v4',]:
                    lrs = [1e-4]
                elif env_id in ['HalfCheetah-v4']:
                    lrs = [1e-3, 1e-4]

                for lr in lrs:


                    ros_lrs = [1e-3, 1e-4]
                    if env_id in ['Swimmer-v4']:
                        ros_lrs = [1e-4, 1e-5]

                    for ros_lr in ros_lrs:
                        for ros_update_epochs in [16]:
                            for ros_mb in [16]:
                                for ros_lambda in [0]:
                                    for ros_num_steps in [256,512, 1024]:
                                        for ros_target_kl in [0.03, 0.05]:
                                            # for ros_anneal_lr in [0, 1]:

                                                if ros_num_steps > num_steps: continue

                                                update_epochs = 10
                                                target_kl = 0.03

                                                ros_clip_coef = 0.3
                                                ros_anneal_lr = 0

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
                                                    ros_mb=ros_mb,
                                                    ros_target_kl=ros_target_kl,
                                                    ros_clip_coef=ros_clip_coef,
                                                    ros_anneal_lr=ros_anneal_lr,
                                                    stats=1,
                                                    ros_lambda=ros_lambda
                                                )
                                                write_to_file(f, args)


if __name__ == "__main__":

    args_hopper()


