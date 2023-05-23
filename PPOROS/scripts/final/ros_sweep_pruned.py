import os

from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file



def args_swimmer():

    env_ids = ['Swimmer-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/s.txt", "w")
            for num_steps in [4096]:
                for lr in [1e-3]:
                    for ros_lr in [1e-5]:
                        for ros_update_epochs in [16]:
                            for ros_lambda in [0]:
                                for ros_num_steps in [4096]:
                                    for ros_target_kl in [0.03]:

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
                                            ros_target_kl=ros_target_kl,
                                            ros_clip_coef=ros_clip_coef,
                                            ros_anneal_lr=ros_anneal_lr,
                                            stats=0,
                                            ros_lambda=ros_lambda
                                        )
                                        write_to_file(f, args)

def args_hopper():

    env_ids = ['Hopper-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/ho.txt", "w")
            for num_steps in [2048]:
                for lr in [1e-4]:
                    for ros_lr in [1e-3]:
                        for ros_update_epochs in [16]:
                            for ros_lambda in [0.1]:
                                for ros_num_steps in [1024, 2048]:
                                    for ros_target_kl in [0.05]:

                                        if ros_num_steps > num_steps: continue
                                        # if ros_target_kl == 0.05 and ros_lambda < 0.5: continue

                                        update_epochs = 10
                                        target_kl = 0.03

                                        ros_clip_coef = 0.3
                                        ros_anneal_lr = 0

                                        for ros_anneal_lr in [0]:
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
                                                stats=0,
                                                ros_lambda=ros_lambda
                                            )
                                            write_to_file(f, args)


def args_halfcheetah():


    env_ids = ['HalfCheetah-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/hc.txt", "w")
            for num_steps in [1024]:
                for lr in [1e-4]:
                    for ros_lr in [1e-4, 1e-5]:
                        for ros_update_epochs in [16]:
                            for ros_lambda in [0.1, 0.5]:
                                for ros_num_steps in [1024]:
                                    for ros_target_kl in [0.03, 0.05]:

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
                                            ros_target_kl=ros_target_kl,
                                            ros_clip_coef=ros_clip_coef,
                                            ros_anneal_lr=ros_anneal_lr,
                                            stats=0,
                                            ros_lambda=ros_lambda
                                        )
                                        write_to_file(f, args)



def args_walker():


    env_ids = ['Walker2d-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/w.txt", "w")
            for num_steps in [2048]:
                for lr in [1e-3]:
                    for ros_lr in [1e-3, 1e-4]:
                        for ros_update_epochs in [16]:
                            for ros_lambda in [0.01, 0.1, 0.5]:
                                for ros_num_steps in [1024, 2048]:
                                    for ros_target_kl in [0.03, 0.05, 0.1]:

                                        if ros_num_steps > num_steps: continue
                                        good = False
                                        if ros_num_steps == 2048:
                                            if ros_lambda in [0.5]:
                                                if ros_target_kl in [0.03]:
                                                    if ros_lr in [1e-4]:
                                                        good = True
                                                if ros_target_kl in [0.05, 0.1]:
                                                    if ros_lr in [1e-3]:
                                                        good = True

                                            if ros_lambda in [0.1]:
                                                if ros_lr in [1e-3]:
                                                    good = True

                                            if ros_lambda in [0.01]:
                                                if ros_target_kl in [0.1]:
                                                    if ros_lr in [1e-4]:
                                                        good = True

                                            if ros_lambda in [0]:
                                                if ros_target_kl in [0.1]:
                                                    if ros_lr in [1e-3]:
                                                        good = True

                                        if ros_num_steps == 1024:
                                            if ros_lambda in [0.5]:
                                                if ros_target_kl in [0.03]:
                                                    good = True

                                            if ros_lambda in [0.1, 0.01]:
                                                if ros_target_kl in [0.1]:
                                                    if ros_lr in [1e-3]:
                                                        good = True
                                                if ros_target_kl in [0.03]:
                                                    if ros_lr in [1e-4]:
                                                        good = True

                                        if not good: continue
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
                                            ros_target_kl=ros_target_kl,
                                            ros_clip_coef=ros_clip_coef,
                                            ros_anneal_lr=ros_anneal_lr,
                                            stats=0,
                                            ros_lambda=ros_lambda
                                        )
                                        write_to_file(f, args)


def args_ant():

    env_ids = ['Ant-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/a.txt", "w")
            for num_steps in [1024]:
                for lr in [1e-4]:
                    for ros_lr in [1e-4, 1e-5]:
                        for ros_update_epochs in [16]:
                            for ros_lambda in [0, 0.01, 0.1,0.5]:
                                for ros_num_steps in [256,512,1024, 2048, 4096, 8192]:
                                    for ros_target_kl in [0.03, 0.05, 0.1]:

                                        if ros_num_steps > num_steps: continue

                                        good = False
                                        if ros_lambda in [0, 0.01, 0.1, 0.5]:
                                            if ros_num_steps in [512]:
                                                if ros_target_kl in [0.03, 0.05]:
                                                    if ros_lr in [1e-4, 1e-5]:
                                                        good = True

                                            if ros_num_steps in [1024]:
                                                if ros_target_kl in [0.05, 0.1]:
                                                    if ros_lr in [1e-5]:
                                                        good = True

                                        if not good: continue

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
                                            ros_target_kl=ros_target_kl,
                                            ros_clip_coef=ros_clip_coef,
                                            ros_anneal_lr=ros_anneal_lr,
                                            stats=0,
                                            ros_lambda=ros_lambda
                                        )
                                        write_to_file(f, args)



def args_humanoid():


    env_ids = ['Humanoid-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/hu.txt", "w")

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

            for ros_num_steps in [1024, 2048, 4096, 8192]:
                for ros_target_kl in [0.03, 0.05, 0.1]:
                    for ros_lambda in [0, 0.01, 0.1,0.5]:

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
                            stats=0,
                            ros_lambda=ros_lambda
                        )
                        write_to_file(f, args)


if __name__ == "__main__":

    args_swimmer()
    args_hopper()
    args_halfcheetah()
    args_walker()
    # args_ant()
    # args_humanoid()

