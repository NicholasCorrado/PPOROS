import os

from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def args_hopper():

    env_ids = ['Humanoid-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [4]:
            f = open(f"commands/ros_b{buffer_size}.txt", "w")
            for s in [8192]:
                for lr in [1e-4,]:
                    for rlr in [1e-4, 1e-5]:
                        for ros_update_epochs in [16]:
                            for ros_mb in [32]:
                                for l in [0.01, 0.1, 0.3]:
                                    for rs in [256, 512, 1024, 2048, 4096, 8192]:
                                        for k in [0.03, 0.05, 0.1]:
                                            # for ros_anneal_lr in [0, 1]:
                                            good = False
                                            if (lr, rlr, s, rs, l, k) in \
                                                [
                                                    (1e-4, 1e-4, 8192, 256, 0.1, 0.1),

                                                    # (1e-4, 1e-4, 8192, 512, 0.3, 0.03),
                                                    # (1e-4, 1e-4, 8192, 512, 0.3, 0.05),
                                                    # (1e-4, 1e-4, 8192, 1024, 0.3, 0.05),
                                                    # (1e-4, 1e-4, 8192, 4096, 0.3, 0.03),
                                                    # (1e-4, 1e-4, 8192, 4096, 0.3, 0.05),
                                                    # (1e-4, 1e-4, 8192, 4096, 0.3, 0.1),
                                                    #
                                                    # (1e-4, 1e-4, 8192, 512, 0.1, 0.03),
                                                    # (1e-4, 1e-4, 8192, 512, 0.1, 0.05),
                                                    # (1e-4, 1e-4, 8192, 1024, 0.1, 0.03),
                                                    #
                                                    # (1e-4, 1e-4, 8192, 512, 0.01, 0.03),
                                                    # (1e-4, 1e-4, 8192, 512, 0.01, 0.05),
                                                    #
                                                    # (1e-3, 1e-4, 2048, 512, 0.01, 0.1),
                                                    # (1e-3, 1e-4, 2048, 1024, 0.01, 0.03),
                                                    # (1e-3, 1e-4, 4096, 2048, 0.01, 0.03),
                                                    # (1e-3, 1e-4, 4096, 2048, 0.01, 0.05),
                                                    #
                                                    # (1e-3, 1e-4, 2048, 512, 0.1, 0.03),
                                                    # (1e-3, 1e-4, 2048, 512, 0.1, 0.05),
                                                    # (1e-3, 1e-4, 2048, 512, 0.1, 0.1),
                                                    #
                                                    # (1e-3, 1e-4, 4096, 2048, 0.1, 0.05),
                                                    # (1e-3, 1e-4, 4096, 2048, 0.1, 0.1),
                                                    # (1e-3, 1e-4, 4096, 2048, 0.3, 0.03),

                                                ]:
                                                    good = True

                                            if not good: continue
                                            if rs > s: continue

                                            update_epochs = 10
                                            target_kl = 0.03

                                            ros_clip_coef = 0.1
                                            ros_anneal_lr = 0

                                            args = gen_args(
                                                device='cpu',
                                                length="short",
                                                arg_generator=ppo_ros,
                                                env_id=env_id,
                                                num_steps=s,
                                                lr=lr,
                                                update_epochs=update_epochs,
                                                target_kl=target_kl,
                                                ros_lr=rlr,
                                                buffer_size=buffer_size,
                                                ros_num_steps=rs,
                                                ros_update_epochs=ros_update_epochs,
                                                ros_mb=ros_mb,
                                                ros_target_kl=k,
                                                ros_clip_coef=ros_clip_coef,
                                                ros_anneal_lr=ros_anneal_lr,
                                                stats=1,
                                                ros_lambda=l
                                            )
                                            write_to_file(f, args)


if __name__ == "__main__":

    args_hopper()


