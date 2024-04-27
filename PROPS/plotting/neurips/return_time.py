import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths, plot

PROPS_PATHS = {
    'Swimmer-v4': {
        2: {
            'path': 'data/rl/props/results/Swimmer-v4/ppo_ros/b_2/s_2048/s_1024/lr_0.001/lr_1e-05/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.3/a_0',
        }
    },
    'HalfCheetah-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/HalfCheetah-v4/ppo_props/b_1/s_1024/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.03/l_0.3/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/HalfCheetah-v4/ppo_ros/b_2/s_1024/s_512/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/l_0.3/e_16/mb_16/c_0.3/a_0',
        }
    },
    'Hopper-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/Hopper-v4/ppo_props/b_1/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.1/l_0.3/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/Hopper-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.05/l_0.3/e_16/mb_16/c_0.3/a_0',
        }
    },
    'Walker2d-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/Walker2d-v4/ppo_props/b_1/s_2048/s_256/lr_0.001/lr_0.0001/kl_0.03/kl_0.05/l_0.1/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/Walker2d-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.1/l_0.3/e_16/mb_32/c_0.3/a_0',
        }
    },
    'Ant-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/Ant-v4/ppo_props/b_1/s_1024/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.03/l_0.1/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/Ant-v4/ppo_ros/b_2/s_2048/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.05/a_0',
        }
    },
    'Humanoid-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/Humanoid-v4/ppo_props/b_1/s_8192/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.03/l_0.3/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/Humanoid-v4/ppo_ros/b_2/s_8192/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.1/l_0.1/e_16/mb_32/c_0.3/a_0',
        },
    },
}

AWPROPS_PATHS = {
    'Swimmer-v4': {
        2: {
            # 'path': 'data/rl/props/results/Swimmer-v4/ppo_ros/b_2/s_2048/s_1024/lr_0.001/lr_1e-05/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.3/a_0',
        }
    },
    'HalfCheetah-v4': {
        2: {
            # 'path': 'data/rl/props/results/HalfCheetah-v4/ppo_ros/b_2/s_1024/s_512/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/l_0.3/e_16/mb_16/c_0.3/a_0',
        }
    },
    'Hopper-v4': {
        2: {
            'path': '/Users/nicholascorrado/code/PROPS/PROPS/plotting/7_04/adv_best/results/Hopper-v4/ppo_props/b_2/s_1024/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.1/l_0.01/c_0.3',
        }
    },
    'Walker2d-v4': {
        2: {
            # 'path': 'data/rl/props/results/Walker2d-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.1/l_0.3/e_16/mb_32/c_0.3/a_0',
        }
    },
    'Ant-v4': {
        2: {
            # 'path': 'data/rl/props/results/Ant-v4/ppo_ros/b_2/s_2048/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.05/a_0',
        }
    },
    'Humanoid-v4': {
        2: {
            # 'path': 'data/rl/props/results/Humanoid-v4/ppo_ros/b_2/s_8192/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.1/l_0.1/e_16/mb_32/c_0.3/a_0',
        },
    },
}


if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')
    env_ids = ['Swimmer-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4']
    env_ids = ['Hopper-v4']

    fig = plt.figure(figsize=(3*3,2*3))
    fig = plt.figure(figsize=(1*4,1*4))

    i = 1


    for env_id in env_ids:
        ax = plt.subplot(2, 3, i)
        ax = plt.subplot(1, 1, i)
        i+=1

        path_dict_all = {}

        time_dict = np.load(f'../rebuttal/times/{env_id}/times.npz')

        ### PROPS ##################################################################################################
        key = rf'PROPS'
        algo = 'ppo_ros'
        results_dir = PROPS_PATHS[env_id][2]['path']
        path_dict_aug = get_paths(
            results_dir=results_dir,
            key=key,
            evaluations_name='evaluations')
        path_dict_aug[key]['times'] = time_dict['PROPS, b=2']
        path_dict_all.update(path_dict_aug)

        ### AW-PROPS ##################################################################################################
        # key = rf'AW-PROPS'
        # algo = 'ppo_ros'
        # try:
        #     results_dir = AWPROPS_PATHS[env_id][2]['path']
        # except:
        #     continue
        # path_dict_aug = get_paths(
        #     results_dir=results_dir,
        #     key=key,
        #     evaluations_name='evaluations')
        # path_dict_all.update(path_dict_aug)

        ### PPO-PRIVILEGED #########################################################################################
        key = rf'PPO-Privileged'
        algo = 'ppo_buffer'
        for s in [1024, 2048, 4096, 8192]:
            for lr in [1e-3, 1e-4]:
                for kl in [0.03]:
                    results_dir = f'data/rl/ppo_privileged/results/{env_id}/{algo}/b_1/s_{s}/lr_{lr}/kl_{kl}/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        x_scale=0.5,
                        evaluations_name='evaluations')
                    path_dict_aug[key]['times'] = time_dict['PPO-Privileged']
                    if len(path_dict_aug[key]['paths']) == 0: continue
                    # path_dict_all.update(path_dict_aug)


        ### PPO-BUFFER #############################################################################################
        key = rf'PPO-Buffer'
        algo = 'ppo_buffer'
        for s in [1024, 2048, 4096, 8192]:
            for lr in [1e-3, 1e-4]:
                for kl in [0.03]:
                    results_dir = f'data/rl/ppo_buffer/results/{env_id}/{algo}/b_2/s_{s}/lr_{lr}/kl_0.03/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        evaluations_name='evaluations')
                    path_dict_aug[key]['times'] = time_dict['PPO-Buffer']
                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all.update(path_dict_aug)

        ### PPO ####################################################################################################
        key = 'PPO'
        algo = 'ppo_buffer'
        for s in [1024, 2048, 4096, 8192]:
            for lr in [1e-3, 1e-4]:
                for kl in [0.03]:
                    results_dir = f'data/rl/ppo_buffer/results/{env_id}/{algo}/b_1/s_{s}/lr_{lr}/kl_0.03/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        evaluations_name='evaluations')
                    path_dict_aug[key]['times'] = time_dict['PPO']

                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all.update(path_dict_aug)

        # key = rf'SAC'
        # algo = 'sac'
        # results_dir = f'../rebuttal/sac_2/results/{env_id}/{algo}/lr_0.0003/lr_0.001/bs_256/a_0'
        # results_dir = f'../rebuttal/sac_hopper/results/{env_id}/{algo}/lr_0.0003/lr_0.001/bs_256/a_0'
        # path_dict_aug = get_paths(
        #     results_dir=results_dir,
        #     key=key,
        #     evaluations_name='evaluations')
        # path_dict_aug[key]['times'] = time_dict['SAC']
        # path_dict_all.update(path_dict_aug)

        plot(path_dict_all, name='returns')
        plt.title(f'{env_id}', fontsize=20)
        plt.xlabel('Time (s)', fontsize=20)
        plt.ylabel('Return', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.xscale('log')

        # Use scientific notation for x-axis
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # set fontsize of 1e6
        ax.xaxis.get_offset_text().set_fontsize(14)

        plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.75)

    # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10)

    save_dir = f'figures'
    save_name = f'return_time.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=600)

    plt.show()
