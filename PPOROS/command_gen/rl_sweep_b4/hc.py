import os

from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def write_args():


    env_ids = ['HalfCheetah-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [4]:
            f = open(f"commands/ros_b{buffer_size}.txt", "w")
            for num_steps in [2048, 4096]:
                for lr in [1e-4]:
                    for ros_lr in [1e-3, 1e-4]:
                        for ros_update_epochs in [16]:
                            for ros_lambda in [0.1, 0.3]:
                                for ros_num_steps in [256, 512]:
                                    for ros_target_kl in [0.03, 0.05]:
                                        for ros_mb in [16]:

                                            # good = False
                                            # if lr == 1e-4 and ros_lr == 1e-3:
                                            #     if ros_lambda == 0.01:
                                            #         if (num_steps, ros_num_steps) == (1024, 1024):
                                            #             if ros_target_kl in [0.1]:
                                            #                 good = True
                                            #         if (num_steps, ros_num_steps) == (4096,4096):
                                            #             if ros_target_kl in [0.05]:
                                            #                 good = True
                                            #     if ros_lambda == 0.1:
                                            #         if (num_steps, ros_num_steps) == (1024, 256):
                                            #             if ros_target_kl in [0.05]:
                                            #                 good = True
                                            #         if (num_steps, ros_num_steps) == (2048, 2048):
                                            #             if ros_target_kl in [0.1]:
                                            #                 good = True
                                            #         if (num_steps, ros_num_steps) == (4096,4096):
                                            #             if ros_target_kl in [0.05]:
                                            #                 good = True
                                            #     if ros_lambda == 0.3:
                                            #         if (num_steps, ros_num_steps) == (1024, 1024):
                                            #             if ros_target_kl in [0.03]:
                                            #                 good = True
                                            #         if (num_steps, ros_num_steps) == (2048, 256):
                                            #             if ros_target_kl in [0.1]:
                                            #                 good = True
                                            #         if (num_steps, ros_num_steps) == (4096,4096):
                                            #             if ros_target_kl in [0.03]:
                                            #                 good = True
                                            # if lr == 1e-4 and ros_lr == 1e-4:
                                            #     if ros_lambda == 0.01:
                                            #         if (num_steps, ros_num_steps) == (1024, 256):
                                            #             if ros_target_kl in [0.03]:
                                            #                 good = True
                                            #         if (num_steps, ros_num_steps) == (2048,2048):
                                            #             if ros_target_kl in [0.1]:
                                            #                 good = True
                                            # if not good: continue

                                            update_epochs = 10
                                            target_kl = 0.03

                                            ros_clip_coef = 0.1
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
                                                ros_target_kl=ros_target_kl,
                                                ros_clip_coef=ros_clip_coef,
                                                ros_anneal_lr=ros_anneal_lr,
                                                ros_mb=ros_mb,
                                                stats=1,
                                                ros_lambda=ros_lambda
                                            )
                                            write_to_file(f, args)
if __name__ == "__main__":

    write_args()


