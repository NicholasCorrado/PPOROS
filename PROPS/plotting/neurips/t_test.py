import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from scipy.stats import stats

from PROPS.plotting.utils import get_paths, plot, load_data

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



def plot(save_dict, name, m=100000, success_threshold=None, return_cutoff=-np.inf, times=None):
    i = 0

    # palette = seaborn.color_palette()
    # print(os.getcwd())

    results = {}

    for agent, info in save_dict.items():
        paths = info['paths']
        x_scale = info['x_scale']
        max_t = info['max_t']
        avgs = []
        for path in paths:
            u, t, avg = load_data(path, name=name, success_threshold=success_threshold)
            if avg is not None:
                # print(len(avg))
                # if len(avg) < 50:
                #     continue
                if max_t:
                    cutoff = np.where(t <= max_t/x_scale)[0]
                    avg = avg[cutoff]
                    t = t[cutoff]

                elif m:
                    avg = avg[:m]
                avgs.append(avg)
                t_good = t

        if len(avgs) == 0:
            continue
        elif len(avgs) == 1:
            avg_of_avgs = avg
            q05 = np.zeros_like(avg)
            q95 = np.zeros_like(avg)

        else:

            min_l = np.inf
            for a in avgs:
                l = len(a)
                if l < min_l:
                    min_l = l

            if min_l < np.inf:
                for i in range(len(avgs)):
                    avgs[i] = avgs[i][:min_l]

            avg_of_avgs = np.mean(avgs, axis=0)


            std = np.std(avgs, axis=0)
            N = len(avgs)
            ci = 1 * std / np.sqrt(N) * 1.96
            q05 = avg_of_avgs - ci
            q95 = avg_of_avgs + ci

            q25 = np.quantile(avgs, 0.25, axis=0)
            q75 = np.quantile(avgs, 0.75, axis=0)
            results[agent] = np.array(avgs)

        i += 1

    return results
    # return fig



if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')
    env_ids = ['Swimmer-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4']
    # env_ids = ['Humanoid-v4']

    fig = plt.figure(figsize=(3*3,2*3))
    i = 1
    results_all = {}
    for env_id in env_ids:
        ax = plt.subplot(2, 3, i)
        i+=1

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
                                            # results_dir = f'../props_b1_50/results/{env_id}/{algo}/b_{b}/s_{s}/s_{rs}/lr_{lr}/lr_{rlr}/kl_0.03/kl_{rkl}/l_{l}/e_16/mb_{ros_mb}/c_0.3/a_0'

                                            # key = rf'PROPS: {s},{rs}; {lr},{rlr}; {rkl}; {l}, {ros_mb}'

                                            path_dict_aug = get_paths(
                                                results_dir=results_dir,
                                                key=key,
                                                evaluations_name='evaluations')
                                            path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])

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
                    if len(path_dict_aug[key]['paths']) == 0: continue
                    path_dict_all.update(path_dict_aug)

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

                    results_dir = f'../more_buffer/results/{env_id}/{algo}/b_2/s_{s}/lr_{lr}/kl_0.03/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        evaluations_name='evaluations')
                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])

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

                    results_dir = f'../more_buffer/results/{env_id}/{algo}/b_1/s_{s}/lr_{lr}/kl_0.03/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        evaluations_name='evaluations')
                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])

        # key = rf'SAC'
        # algo = 'sac'
        # try:
        #     # results_dir = f'../rebuttal/sac_2/results/{env_id}/{algo}/lr_0.0003/lr_0.001/bs_256/a_0'
        #     if env_id in ['Hopper-v4']:
        #         results_dir = f'../rebuttal/sac_hopper/results/{env_id}/{algo}/lr_0.0003/lr_0.001/bs_256/a_0'
        #     elif env_id in ['HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4']:
        #         results_dir = f'../rebuttal/sac_larger_budget/results/{env_id}/{algo}/lr_0.001/lr_0.003/bs_256/a_0'
        #     else:
        #         results_dir = f'../rebuttal/sac_2/results/{env_id}/{algo}/lr_0.0003/lr_0.001/bs_256/a_0'
        #     path_dict_aug = get_paths(
        #         results_dir=results_dir,
        #         key=key,
        #         evaluations_name='evaluations')
        #     path_dict_all.update(path_dict_aug)
        # except:
        #     pass

        results = plot(path_dict_all, name='returns')
        results_all[env_id] = results

    for env_id, results in results_all.items():
        print(env_id)
        ppo = results['PPO']
        ppo_buffer = results['PPO-Buffer']
        props = results['PROPS']
        ppo_priv = results['PPO-Privileged']

        l = np.min([len(ppo), len(props)])
        '''
        s 1 1
        ho 2 1
        hc 4 1
        W 1 2
        a 2 1
        hu 1 1
        '''
        l = np.min([len(ppo_buffer), len(props), 50])

        # k = np.min([ppo_buffer.shape[-1], props.shape[-1]])
        # if env_id in ['Swimmer-v4', 'Humanoid-v4']:
        m, n = ppo_buffer.shape[-1], props.shape[-1]

        N = 3
        print('PPO-BUFFER/PROPS')
        for i in range(1, N+1):
            print(stats.ttest_rel(ppo_buffer[:l, m//N*i-1], props[:l, n//N*i-1]), ppo_buffer[:l, m//N*i-1].mean(), props[:l, n//N*i-1].mean())

        m, n = ppo.shape[-1], props.shape[-1]

        print('PPO/PROPS')
        for i in range(1, N + 1):
            print(stats.ttest_rel(ppo[:l, m//N*i-1], props[:l, n//N*i-1]))

    for env_id, results in results_all.items():
        print(env_id)
        ppo = results['PPO']
        ppo_buffer = results['PPO-Buffer']
        props = results['PROPS']
        ppo_priv = results['PPO-Privileged']
        i = 1
        m, n, o, p = props.shape[-1], ppo_priv.shape[-1], ppo_buffer.shape[-1], ppo.shape[-1]
        l = 50
        # print(int(props[:l,  m// N * i - 1].mean()), int(ppo_priv[:l,  n// N * i - 1].mean()), int(ppo_buffer[:l, o//N*i-1].mean()), int(ppo[:l, p// N * i - 1].mean()), )
        # print(int(props[:l,  m// N * i - 1].mean()), int(ppo_buffer[:l, o//N*i-1].mean()), int(ppo[:l, p// N * i - 1].mean()), )
        print(props[:l,  m// N * i - 1].mean(), ppo_buffer[:l, o//N*i-1].mean(), ppo[:l, p// N * i - 1].mean(), )