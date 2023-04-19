from PPOROS.plotting.condor_no_anneal.ppo import BEST_PPO_PARAMS
from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def write_args():


    # env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Ant-v4', 'Walker2d-v4', 'InvertedPendulum-v4', 'InvertedDoublePendulum-v4', ]
    env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Ant-v4', 'Walker2d-v4', 'InvertedDoublePendulum-v4', ]

    # env_ids = ['Humanoid-v4', 'Swimmer-v4', ]

    # env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    # env_ids = ['Hopper-v4']
    # env_ids = ['Ant-v4', 'Humanoid-v4']


    env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4', ]
    # env_ids = ['Walker2d-v4', 'HalfCheetah-v4',] #'Ant-v4', 'Humanoid-v4', 'Swimmer-v4', ]
    # env_ids = ['Hopper-v4']

    # env_ids = ['Ant-v4', 'Humanoid-v4', 'Swimmer-v4',]

    # env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4']

    # env_ids = ['Walker2d-v4']

    # env_ids = ['HalfCheetah-v4', 'Walker2d-v4', 'Hopper-v4',]# 'Ant-v4', 'Humanoid-v4',  'Swimmer-v4']
    # env_ids = ['Ant-v4', 'Humanoid-v4', 'Swimmer-v4',]

    env_ids = ['Hopper-v4', 'Walker2d-v4']
    env_ids = ['Ant-v4']

    # for env_id in env_ids:
    #     for lr in [1e-3, 1e-4]:
    #         for target_kl in [0.03]:
    #             for ros_history in [2,4,8]:
    #                 args = gen_args(
    #                     device='cpu',
    #                     length="short",
    #                     arg_generator=ppo_buffer,
    #                     env_id=env_id,
    #                     lr=lr,
    #                     target_kl=target_kl,
    #                     num_steps=num_steps,
    #                     ros_history=ros_history,
    #                     stats=0,
    #                 )
    #                 write_to_file(f, args)
    #
    # for env_id in env_ids:
    #     for lr in [1e-3, 1e-4]:
    #         for target_kl in [0.03]:
    #             for df in [1,2,4,8]:
    #                 args = gen_args(
    #                     device='cpu',
    #                     length="short",
    #                     arg_generator=ppo,
    #                     env_id=env_id,
    #                     lr=lr,
    #                     target_kl=target_kl,
    #                     num_steps=int(num_steps*df),
    #                     total_timesteps=int(TIMESTEPS[env_id]*df),
    #                     stats=0,
    #                 )
    #                 write_to_file(f, args)

    # ros_target_kl = 0.05
    # for ros_history in [4]:
    #     for num_steps in [1024, 2048, 4096, 8192]:
    #         for env_id in env_ids:
    #             for lr in [1e-3, 1e-4]:
    #                 for lr_ros in [1e-5]:
    #                     for ros_update_epochs in [16,32]:
    #                         args = gen_args(
    #                             device='cpu',
    #                             length="short",
    #                             arg_generator=ppo_ros,
    #                             env_id=env_id,
    #                             lr=lr,
    #                             lr_ros=lr_ros,
    #                             num_steps=num_steps,
    #                             ros_history=ros_history,
    #                             ros_update_epochs=ros_update_epochs,
    #                             ros_target_kl=ros_target_kl,
    #                             stats=0,
    #                         )
                            # write_to_file(f, args)

    for env_id in env_ids:
        f = open(f"args/ros_b24_{env_id}.txt", "w")
        for ros_history in [2,4]:
            num_steps_list = [1024, 2048, 4096]
            if env_id in ['Ant-v4']:
                num_steps_list = [1024, 2048, 4096, 8192]
            if env_id in ['Humanoid-v4']:
                num_steps_list = [4096, 8192]
            for num_steps in num_steps_list:
                ros_num_steps = 256
                for lr in [1e-3, 1e-4]:
                    for lr_ros in [1e-3, 1e-4]:
                        for ros_clip_coef in [0.2]:
                            for ros_update_epochs in [5, 10]:
                                for update_epochs in [10]:
                                    target_kl = BEST_PPO_PARAMS[env_id]['target_kl']
                                    for ros_target_kl in [0.05]:
                                        args = gen_args(
                                            device='cpu',
                                            length="short",
                                            arg_generator=ppo_ros,
                                            env_id=env_id,
                                            num_steps=num_steps,
                                            lr=lr,
                                            update_epochs=update_epochs,
                                            target_kl=target_kl,
                                            lr_ros=lr_ros,
                                            ros_history=ros_history,
                                            ros_num_steps=ros_num_steps,
                                            ros_update_epochs=ros_update_epochs,
                                            ros_target_kl=ros_target_kl,
                                            ros_clip_coef=ros_clip_coef,
                                            stats=0,
                                        )
                                        write_to_file(f, args)


if __name__ == "__main__":

    write_args()


