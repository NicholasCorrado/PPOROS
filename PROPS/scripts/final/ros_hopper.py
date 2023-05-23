import os

from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def args_hopper():

    env_ids = ['Hopper-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2, 4]:
            f = open(f"commands/ros_b{buffer_size}.txt", "w")
            for num_steps in [1024, 2048]:
                for lr in [1e-4]:
                    for ros_lr in [1e-3, 1e-4]:
                        for ros_update_epochs in [16]:
                            for ros_mb in [16, 32]:
                                for ros_lambda in [0.1]:
                                    for ros_num_steps in [64]:
                                        for ros_target_kl in [0.01]:
                                            # for ros_anneal_lr in [0, 1]:

                                                if ros_num_steps > num_steps: continue
                                                # if ros_target_kl == 0.05 and ros_lambda < 0.5: continue
                                                #
                                                # good = False
                                                # if ros_lambda in [0.5]:
                                                #     if ros_mb in [8, 32] and ros_anneal_lr == 0:
                                                #         good = True
                                                #     if ros_mb in [16]:
                                                #         if ros_target_kl in [0.1] and ros_anneal_lr == 1:
                                                #             good = True
                                                #
                                                # if ros_lambda in [0.1]:
                                                #     if ros_anneal_lr == 1:
                                                #         good = True
                                                #     if ros_mb in [8, 32]:
                                                #         if ros_target_kl in [0.1] and ros_anneal_lr == 0:
                                                #             good = True
                                                #
                                                # if ros_lambda in [0.01]:
                                                #     if ros_target_kl in [0.1] and ros_mb==32 and ros_anneal_lr == 1:
                                                #         good = True
                                                #     if ros_target_kl in [0.05] and ros_mb==8 and ros_anneal_lr == 0:
                                                #         good = True
                                                #     if ros_mb in [16]:
                                                #         good = True
                                                #
                                                # if not good: continue
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
                                                    stats=0,
                                                    ros_lambda=ros_lambda
                                                )
                                                write_to_file(f, args)


if __name__ == "__main__":

    args_hopper()


