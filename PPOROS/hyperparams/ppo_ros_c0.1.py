"""

"""



PPOROS_PARAMS = {
    'Swimmer-v4': {
        2: {
            # 'path': 'condor_new/ppo_ros_b2_swimmer_final/results/Swimmer-v4/ppo_ros/history_2/num_steps_4096/num_steps_256/lr_0.001/lr_1e-05/kl_0.05/kl_0.05/epochs_10/epochs_8/clip_0.3/',
            'path': '5_08/whcho_final/results/Swimmer-v4/ppo_ros/b_2/s_4096/s_2048/lr_0.001/lr_0.0001/kl_0.03/kl_0.03/l_0.01/e_16/mb_16/c_0.3/a_0',
            'num_steps': 4096,
            'ros_num_steps': 512,
            'lr': 1e-3,
            'ros_lr': 1e-5,
            'ros_lambda': 0.01,
            'ros_target_kl': 0.1
        },
        4: {
        },
        8: {
        },
    },
    'HalfCheetah-v4': {
        2: {
            # 'path': 'condor_fix/fix_kl_hwH/results/HalfCheetah-v4/ppo_ros/b_2/l_0.01/s_1024/s_1024/lr_0.0001/lr_0.0001/kl_0.03/kl_0.1/e_16/c_0.3/a_1/',
            'path': '5_08/whcho_final/results/HalfCheetah-v4/ppo_ros/b_2/s_1024/s_512/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/l_0.3/e_16/mb_16/c_0.3/a_0',
            'num_steps': 1024,
            'ros_num_steps': 1024,
            'lr': 1e-3,
            'ros_lr': 1e-3,
            'ros_lambda': 0.01,
            'ros_target_kl': 0.1
        },
        4: {

        },
        8: {

        },
    },
    'Hopper-v4': {
        2: {
            #'path': 'condor_fix/fix_kl/results/Hopper-v4/ppo_ros/b_2/l_0.5/s_2048/s_2048/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/e_16/c_0.3/a_1/',
            # 'path': './r_30/ho/results/Hopper-v4/ppo_ros/b_2/s_2048/s_512/lr_0.0001/lr_0.001/kl_0.03/kl_0.1/l_0.01/e_16/mb_16/c_0.3/a_0',
            'path': '5_08/r50/results/Hopper-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.05/l_0.3/e_16/mb_16/c_0.3/a_0',
            'num_steps': 2048,
            'lr': 1e-4,
            'ros_num_steps': 1024,
            'ros_lr': 1e-3,
            'ros_lambda': 0.01,
            'ros_target_kl': 0.1,
        },
        4: {
        },
        8: {
        },
    },
    'Walker2d-v4': {
        2: {
            # 'path': './condor_fix/fix_kl_0.001/results/Walker2d-v4/ppo_ros/b_2/l_0.001/s_2048/s_256/lr_0.001/lr_0.0001/kl_0.03/kl_0.1/e_16/c_0.3/a_0',
            # 'path': './r_30/w/results/Walker2d-v4/ppo_ros/b_4/s_2048/s_512/lr_0.001/lr_0.0001/kl_0.03/kl_0.03/l_0.01/e_16/mb_16/c_0.3/a_0',
            # 'path': './r_30/w/results/Walker2d-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.3/a_0',
            'path': '5_08/r50/results/Walker2d-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.1/l_0.3/e_16/mb_32/c_0.3/a_0',
            # 'path': '5_08/r50/results/Walker2d-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.1/l_0.3/e_16/mb_16/c_0.3/a_0',
            'num_steps': 2048,
            'lr': 1e-3,
            'ros_num_steps': 2048,
            'ros_lr': 1e-4,
            'ros_lambda': 0.1,
            'ros_target_kl': 0.1,
        },
        4: {
            'path': '5_06/ros_b4_1/results/Walker2d-v4/ppo_ros/b_4/s_2048/s_256/lr_0.001/lr_0.0001/kl_0.03/kl_0.03/l_0.01/e_16/mb_32/c_0.3/a_0',
            'num_steps': 1024,
            'lr': 1e-3,
            'ros_num_steps': 1024,
            'ros_lr': 1e-3,
            'ros_lambda': 0.1,
            'ros_target_kl': 0.1,
        },
        8: {
        },
    },
    'Ant-v4': {
        2: {
            # 'path': './condor_sweep/a_10/results/Ant-v4/ppo_ros/b_2/l_0.01/s_1024/s_512/lr_0.0001/lr_0.001/kl_0.03/kl_0.1/e_16/c_0.3/a_1/',
            # 'path': './condor_sweep/a_10/results/Ant-v4/ppo_ros/b_2/l_0.01/s_1024/s_512/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/e_16/c_0.3/a_1/',
            # 'path': './r_30/a/results/Ant-v4/ppo_ros/b_2/s_1024/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.03/l_0.1/e_16/mb_32/c_0.3/a_0',
            # 'path': './5_06/ros_50_2/results/Ant-v4/ppo_ros/b_2/s_2048/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.1/l_0.1/e_16/mb_32/c_0.3/a_0',
            # 'path': '5_08/a/results/Ant-v4/ppo_ros/b_2/s_1024/s_1024/lr_0.0001/lr_0.0001/kl_0.03/kl_0.1/l_0.1/e_16/mb_16/c_0.3/a_0',
            'path': '5_09/a_final/results/Ant-v4/ppo_ros/b_2/s_1024/s_1024/lr_0.0001/lr_0.0001/kl_0.03/kl_0.1/l_0.1/e_16/mb_16/c_0.3/a_0',
            'num_steps': 1024,
            'lr': 1e-4,
            'ros_num_steps': 512,
            'ros_lr': 1e-3,
            'ros_lambda': 0.01,
            'ros_target_kl': 0.05,
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
