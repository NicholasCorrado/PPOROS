"""

"""



PPOROS_PARAMS = {
    'Swimmer-v4': {
        2: {
            'path': '5_12/ros_b2_c0.1/results/Swimmer-v4/ppo_ros/b_2/s_2048/s_512/lr_0.001/lr_0.0001/kl_0.03/kl_0.03/l_0.3/e_16/mb_16/c_0.1/a_0',
        },
        3: {
            'path': '5_12/ros_b2_c0.1/results/Swimmer-v4/ppo_ros/b_2/s_2048/s_512/lr_0.001/lr_0.001/kl_0.03/kl_0.03/l_0.3/e_16/mb_16/c_0.1/a_0',
        },
        4: {
            'path': '5_12/ros_b2_c0.1/results/Swimmer-v4/ppo_ros/b_2/s_2048/s_512/lr_0.001/lr_0.001/kl_0.03/kl_0.03/l_0.3/e_16/mb_16/c_0.1/a_0',
        },
        8: {
        },
    },
    'HalfCheetah-v4': {
        2: {
            'path': 'PPOROS/plotting/5_12/ros_b2_c0.1/results/HalfCheetah-v4/ppo_ros/b_2/s_2048/s_512/lr_0.0001/lr_0.001/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.1/a_0',
        },
        2: {
            'path': 'PPOROS/plotting/5_12/ros_b2_c0.1/results/HalfCheetah-v4/ppo_ros/b_2/s_2048/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/l_0.1/e_16/mb_16/c_0.1/a_0',
        },
        4: {

        },
        8: {

        },
    },
    'Hopper-v4': {
        2: {
            'path': '5_12/ros_b2_c0.1/results/Hopper-v4/ppo_ros/b_2/s_1024/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.03/l_0.3/e_16/mb_16/c_0.1/a_0',
        },
        3: {
            'path': '5_12/ros_b2_c0.1/results/Hopper-v4/ppo_ros/b_2/s_1024/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/l_0.1/e_16/mb_16/c_0.1/a_0',
        },
        4: {
        },
        8: {
        },
    },
    'Walker2d-v4': {
        2: {
            'path': '5_12/ros_b2_c0.1/results/Walker2d-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.0001/kl_0.03/kl_0.05/l_0.3/e_16/mb_16/c_0.1/a_0',
        },
        2: {
            'path': '5_12/ros_b2_c0.1/results/Walker2d-v4/ppo_ros/b_2/s_2048/s_512/lr_0.001/lr_0.0001/kl_0.03/kl_0.03/l_0.3/e_16/mb_16/c_0.1/a_0',
        },
        4: {
            'path': '5_06/ros_b4_1/results/Walker2d-v4/ppo_ros/b_4/s_2048/s_256/lr_0.001/lr_0.0001/kl_0.03/kl_0.03/l_0.01/e_16/mb_32/c_0.3/a_0',
        },
        8: {
        },
    },
    'Ant-v4': {
        2: {
            'path': '5_12/ros_b2_c0.1/results/Ant-v4/ppo_ros/b_2/s_1024/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.03/l_0.3/e_16/mb_16/c_0.1/a_0',
        },
        3: {
            'path': '5_12/ros_b2_c0.1/results/Ant-v4/ppo_ros/b_2/s_1024/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.05/l_0.1/e_16/mb_16/c_0.1/a_0',
        },
        4: {
        },
        8: {
        },
    },
    'Humanoid-v4': {
        2: {
            # 'path': './condor_X/X_ros_humanoid_b2_sweep/results/Humanoid-v4/ppo_ros/b_2/s_8192/lr_0.0001/lr_0.0001/kl_0.03/kl_0.02/e_16/c_0.2/',
            # 'path': 'condor_fix/fix_kl_hwH/results/Humanoid-v4/ppo_ros/b_2/l_0.5/s_8192/s_8192/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/e_16/c_0.3/a_1/',
            # 'path': './r_30/hu/results/Humanoid-v4/ppo_ros/b_2/s_8192/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.03/l_0.1/e_16/mb_32/c_0.3/a_0',
            'path': './5_06/ros_50_2/results/Humanoid-v4/ppo_ros/b_2/s_8192/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.1/l_0.1/e_16/mb_32/c_0.3/a_0',
            'num_steps': 8192,
            'ros_num_steps': 8192,
            'lr': 1e-4,
            'ros_lr': 1e-3,
            'ros_lambda': 0.1,
            'ros_target_kl': 0.05, # 0.1
        },
        4: {
        },
        8: {
        },
    },
}
