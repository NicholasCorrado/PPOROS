import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths, plot, load_data

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from rliable.plot_utils import _decorate_axis, _annotate_and_decorate_axis



def load_data(paths, name, success_threshold=None):
    avgs = []
    for path in paths:
        with np.load(path) as data:
            # for key in data:
            #     print(key)

            try:
                t = data['t']
            except:
                t = data['timesteps']

            if name == 'se_normalized':
                r = np.clip(data['diff_kl_mle_target']/np.abs(data['ref_kl_mle_target']), -10000, 100000)
            elif name == 'diff_kl_mle_target':
                r = np.clip(data['diff_kl_mle_target'], -10000,10000)
            else:
                r = data[name]
            if success_threshold is not None:
                r = r > success_threshold

            if len(r.shape) > 1:
                avg = np.average(r, axis=1)
            else:
                avg = r

            avgs.append(avg)

    return t, np.array(avgs)

def get_data(path_dict):
  results = {}
  timesteps = {}
  for agent, info in path_dict.items():
    paths = info['paths']

    t, avgs = load_data(paths, name='returns')
    results[agent] = avgs

  return results

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
            # 'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/iclr/a_b2/results/Ant-v4/ppo_props/b_2/s_2048/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.03/l_0.01/c_0.3',
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/iclr/a_b2/results/Ant-v4/ppo_props/b_2/s_1024/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.03/l_0.3/c_0.3',
            'path2': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/iclr/a_b2/results/Ant-v4/ppo_props/b_2/s_1024/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/l_0.3/c_0.3',
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
    # env_ids = ['Ant-v4']


    fig, ax = plt.subplots(1,1, figsize=(6,6))
    i = 0

    aggr_dict = {'PROPS': [], 'PPO-Buffer': [], 'PPO': []}
    for env_id in env_ids:

        path_dict_all = {}
        print(env_id)
        ### PROPS ##################################################################################################
        key = rf'PROPS'
        algo = 'ppo_ros'
        try:
            results_dir = PROPS_PATHS[env_id][2]['path']
            path_dict_aug = get_paths(
                results_dir=results_dir,
                key=key,
                evaluations_name='evaluations')
            path_dict_all.update(path_dict_aug)
        except:
            pass

        algo = 'ppo_ros'
        ns = [1024, 2048, 4096, 8192]
        lrs = [1e-3, 1e-4]
        ros_lrs = [1e-3, 1e-4, 1e-5]

        for b in [2]:
            for s in ns:
                for lr in lrs:
                    for rlr in ros_lrs:
                        for ros_update_epochs in [16]:
                            for ros_mb in [16, 32]:
                                for l in [0.01, 0.1, 0.3]:
                                    for rs in [256, 512, 1024]:
                                        for rkl in [0.03, 0.05, 0.1]:

                                            results_dir = f'../more/results/{env_id}/{algo}/b_{b}/s_{s}/s_{rs}/lr_{lr}/lr_{rlr}/kl_0.03/kl_{rkl}/l_{l}/e_16/mb_{ros_mb}/c_0.3/a_0'
                                            # results_dir = f'../more_2/results/{env_id}/{algo}/b_{b}/s_{s}/s_{rs}/lr_{lr}/lr_{rlr}/kl_0.03/kl_{rkl}/l_{l}/e_16/mb_{ros_mb}/c_0.3/a_0'
                                            results_dir = f'../props_b1_50/results/{env_id}/{algo}/b_{b}/s_{s}/s_{rs}/lr_{lr}/lr_{rlr}/kl_0.03/kl_{rkl}/l_{l}/e_16/mb_{ros_mb}/c_0.3/a_0'

                                            # key = rf'PROPS: {s},{rs}; {lr},{rlr}; {rkl}; {l}, {ros_mb}'

                                            path_dict_aug = get_paths(
                                                results_dir=results_dir,
                                                key=key,
                                                evaluations_name='evaluations')
                                            path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])

        try:
            results_dir = PROPS_PATHS[env_id][2]['path2']
            path_dict_aug = get_paths(
                results_dir=results_dir,
                key=key,
                evaluations_name='evaluations')
            if len(path_dict_aug[key]['paths']) > 0:
                path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])
        except:
            pass

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
        # key = rf'PPO-Privileged'
        # algo = 'ppo_buffer'
        # for s in [1024, 2048, 4096, 8192]:
        #     for lr in [1e-3, 1e-4]:
        #         for kl in [0.03]:
        #             results_dir = f'data/rl/ppo_privileged/results/{env_id}/{algo}/b_1/s_{s}/lr_{lr}/kl_{kl}/e_10'
        #
        #             path_dict_aug = get_paths(
        #                 results_dir=results_dir,
        #                 key=key,
        #                 x_scale=0.5,
        #                 evaluations_name='evaluations')
        #             if len(path_dict_aug[key]['paths']) == 0: continue
        #             path_dict_all.update(path_dict_aug)
        #

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
                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all.update(path_dict_aug)

                    # results_dir = f'../more_buffer/results/{env_id}/{algo}/b_2/s_{s}/lr_{lr}/kl_0.03/e_10'
                    #
                    # path_dict_aug = get_paths(
                    #     results_dir=results_dir,
                    #     key=key,
                    #     evaluations_name='evaluations')
                    # if len(path_dict_aug[key]['paths']) > 0:
                    #     path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])


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
                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all.update(path_dict_aug)
                    #
                    # results_dir = f'../more_buffer/results/{env_id}/{algo}/b_1/s_{s}/lr_{lr}/kl_0.03/e_10'
                    #
                    # path_dict_aug = get_paths(
                    #     results_dir=results_dir,
                    #     key=key,
                    #     evaluations_name='evaluations')
                    # if len(path_dict_aug[key]['paths']) > 0:
                    #     path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])

        ### SAC ####################################################################################################

        # key = f'SAC'
        # # for (alr, clr) in [(1e-3, 3e-3), (3e-4, 1e-3)]:
        # if env_id in ['Walker2d-v4']:
        #     alr, clr = 1e-3, 3e-3
        # else:
        #     alr, clr = 3e-4, 1e-3
        # results_dir = f'../iclr/sac/results/{env_id}/sac/lr_{alr}/lr_{clr}/bs_256/a_0'
        # path_dict_aug = get_paths(
        #     results_dir=results_dir,
        #     key=key,
        #     evaluations_name='evaluations')
        # if len(path_dict_aug[key]['paths']) > 0:
        #     path_dict_all.update(path_dict_aug)

        # plot(path_dict_all, name='returns')
        # plt.title(f'{env_id}', fontsize='xx-large')
        # plt.xlabel('Timestep', fontsize='xx-large')
        # plt.ylabel('Return', fontsize=20)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)

        ### GePPO ##################################################################################################

        # for n in [1024, 2048]:
        #     # for lr in [1e-3, 1e-4]:
        #     #     for B in [2]:
        #             for M in [2]:
        #                 if 'Swim' in env_id:
        #                     lr = 1e-3
        #                     B=2
        #                 if 'Hop' in env_id:
        #                     lr = 1e-4
        #                     B=1
        #                 if 'Walk' in env_id:
        #                     lr = 1e-4
        #                     B=2
        #                 if 'Half' in env_id:
        #                     lr = 1e-3
        #                     B=2
        #
        #                 key = rf'GePPO'
        #                 algo = 'geppo'
        #                 root_dir = '/Users/nicholascorrado/code/tmp/GePPO/geppo/' + f'condor/results/{env_id}/geppo'
        #                 results_dir = f'{root_dir}/n_{n}/lr_{lr}/B_{B}/M_{M}/v_1/a_1'
        #                 print(results_dir)
        #                 path_dict_aug = get_paths(
        #                     results_dir=results_dir,
        #                     key=key,
        #                     evaluations_name='evaluations')
        #                 if len(path_dict_aug[key]['paths']) > 0:
        #                     path_dict_all.update(path_dict_aug)
        #

        x_scale = {
            'Swimmer-v4': 2,
            'Hopper-v4': 2,
            'HalfCheetah-v4': 4,
            'Walker2d-v4': 2,
            'Ant-v4': 4,
            'Humanoid-v4': 2
        }
        max_thresholds = {
            'Swimmer-v4': 0,
            'Hopper-v4': 0,
            'HalfCheetah-v4': 0,
            'Walker2d-v4': 0,
            'Ant-v4': 0,
            'Humanoid-v4': 0
        }

        return_dict = get_data(path_dict_all)
        for k, v in return_dict.items():
            m = np.max(v)
            if m > max_thresholds[env_id]:
                max_thresholds[env_id] = m

        ts = np.linspace(0, 1, 11)


        for k, v in return_dict.items():
            indices = (ts * (len(v[0,:])-1)).astype(int)
            aggr_dict[k].append(v[:, indices]/max_thresholds[env_id])

    timestep_dict = {}
    for k, v in aggr_dict.items():
        aggr_dict[k] = np.concatenate(v)
        timestep_dict[k] = ts

    thresholds = np.linspace(0.0, 1, 101)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        aggr_dict, thresholds, reps=5000)

    algorithms = list(aggr_dict.keys())
    # Plot score distributions
    # fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        score_distributions, thresholds,
        performance_profile_cis=score_distributions_cis,
        colors=dict(zip(algorithms, seaborn.color_palette('colorblind'))),
        xlabel=r'Return $(\tau)$',
        # linestyles=linestyles,
        # alpha=0,
        ax=ax,
    )
    plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5)
    plt.ylabel(r'Fraction of runs with score > $\tau$', fontsize='xx-large')
    plt.xlabel(r'Normalized Return $(\tau)$', fontsize='xx-large')


    # plt.tight_layout()
    # plt.show()

    plt.title('Aggregate Performance')

    # fig.supxlabel('IQM Return', fontsize='large')

    plt.tight_layout()
    # Push plots down to make room for the the legend
    # fig.subplots_adjust(top=0.88)
    #
    # # Fetch and plot the legend from one of the subplots.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=1, fontsize='xx-large')
    #
    save_dir = f'figures'
    save_name = f'performance_profile.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=300)
    #

    plt.show()
