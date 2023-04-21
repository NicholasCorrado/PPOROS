import os

from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def write_args():


    env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4', ]
    # env_ids = ['Hopper-v4', 'Walker2d-v4']
    # env_ids = ['Humanoid-v4']
    env_ids = ['Hopper-v4', 'Walker2d-v4', 'Swimmer-v4']
    # env_ids = ['Swimmer-v4']
    # env_ids = ['Ant-v4']
    # env_ids = ['Hopper-v4']
    # env_ids = ['Humanoid-v4']


    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [4]:
            f = open(f"commands/ros_b{buffer_size}.txt", "w")
            num_steps_list = [1024, 2048, 4096, 8192]
            if env_id in ['Hopper-v4', 'Walker2d-v4']:
                num_steps_list = [1024, 2048]
            if env_id in ['Swimmer-v4']:
                num_steps_list = [1024, 2048, 4096]
            if env_id in ['Humanoid-v4']:
                num_steps_list = [8192]
            for num_steps in num_steps_list:
                ros_num_steps = 256
                for lr in [1e-3, 1e-4]:
                    if env_id in ['Humanoid-v4'] and lr==1e-3: continue
                    for ros_lr in [1e-3, 1e-4]:
                        for ros_clip_coef in [0.2,0.3]:
                            for ros_update_epochs in [4,8,16]:
                                for update_epochs in [10]:
                                    for target_kl in [0.03]:
                                        for ros_target_kl in [0.02, 0.05]:
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
                                                stats=0,
                                                anneal_lr=1,
                                                log_stats=1
                                            )
                                            write_to_file(f, args)


if __name__ == "__main__":

    write_args()


