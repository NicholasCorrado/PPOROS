import itertools
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths


def load_data(path, successes=False, success_threshold=None):
    with np.load(path, allow_pickle=False) as data:
        try:
            t = data['t']
        except:
            t = data['timesteps']
        try:
            avg = data['kl_mle_target']
        except:
            avg = data['sampling_error']

    return t, avg


def plot(save_dict, use_successes, updates=True, m=None, max_t=None, success_threshold=None):
    i = 0

    print(os.getcwd())
    palette = itertools.cycle(seaborn.color_palette())

    for agent, info in save_dict.items():
        paths = info['paths']
        x_scale = info['x_scale']
        avgs = []
        for path in paths:
            t_tmp, avg = load_data(path, successes=use_successes, success_threshold=success_threshold)
            if avg is not None:
                avgs.append(avg)
                t = np.array(t_tmp) * x_scale

        if m is not None:
            avgs = np.array(avgs)[:, :m]
            t = t[:m]

        if len(avgs) == 0:
            continue
        elif len(avgs) == 1:
            avg_of_avgs = avg
            q05 = np.zeros_like(avg)
            q95 = np.zeros_like(avg)

        else:
            avg_of_avgs = np.average(avgs, axis=0)
            std = np.std(avgs, axis=0)
            N = len(avgs)
            ci = std / np.sqrt(N)
            q05 = avg_of_avgs - ci
            q95 = avg_of_avgs + ci

        style_kwargs = {'linewidth': 1.5}
        color = None
        if 'PPO' in agent:
            style_kwargs['color'] = 'k'
            style_kwargs['linestyle'] = ':'
            style_kwargs = {'linewidth': 1.5}

        if 'PROPS' == agent:
            style_kwargs['color'] = 'k'
            style_kwargs['linestyle'] = ':'
            style_kwargs = {
                'linewidth': 3,
            }

        if 'PROPS, no clip, no reg' in agent:
            style_kwargs['color'] = 'k'
            style_kwargs['linestyle'] = ':'
            color = next(palette)
            style_kwargs = {
                'linewidth': 3,
                'linestyle': '--',
                # 'color': next(palette)
            }

        if 'Buffer' in agent:
            style_kwargs['linestyle'] = '--'

        plt.plot(t, avg_of_avgs, label=agent, **style_kwargs, color=color)
        print(f'{agent}, {q05[1]:.3f}, {q95[1]:.3f}')
        plt.fill_between(t, q05, q95, alpha=0.2, color=color)

        i += 1


if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')
    env_ids = ['Swimmer-v4', 'Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4']
    env_ids = ['Walker2d-v4']

    for policy in ['expert', 'random']:
        subplot_i = 1
        fig = plt.figure(figsize=(1* 3.7, 3.7))

        for env_id in env_ids:
            path_dict_all = {}

            root_dir = f'./data/se_fixed/results/{env_id}'

            algo = 'ppo_ros'
            if env_id in ['Swimmer-v4', 'Ant-v4', 'Humanoid-v4']:
                lr = 1e-4
            else:
                lr = 1e-3
            if env_id in ['Ant-v4', 'Humanoid-v4']:
                l = 0.3
            else:
                l = 0.1

            key = rf'PROPS'
            algo = 'ppo_ros'
            path_dict_aug = get_paths(
                results_dir=f'{root_dir}/{algo}/{policy}/b_16/s_1024/s_256/lr_{lr}/l_{l}/kl_0.05',
                key=key,
                x_scale=1 / 256,
                evaluations_name='stats')
            path_dict_all.update(path_dict_aug)

            key = rf'PROPS, no clip, no reg'
            algo = 'ppo_ros'
            path_dict_aug = get_paths(
                results_dir=f'data/se_fixed_ablation/results/{env_id}/{algo}/{policy}/b_16/no_clip_lambda',
                key=key,
                x_scale=1 / 256,
                evaluations_name='stats')
            path_dict_all.update(path_dict_aug)

            key = rf'ROS'
            algo = 'ros'
            ros_lr = 1e-5
            path_dict_aug = get_paths(
                results_dir=f'{root_dir}/{algo}/{policy}/b_16/s_1024/lr_{ros_lr}',
                key=key,
                x_scale=1 / 256,
                evaluations_name='stats')
            path_dict_all.update(path_dict_aug)

            algo = 'ppo_buffer'
            key = rf'OS'
            path_dict_aug = get_paths(
                results_dir=f'{root_dir}/{algo}/{policy}/b_16',
                key=key,
                x_scale=1 / 256,
                evaluations_name='stats')
            path_dict_all.update(path_dict_aug)

            plot(path_dict_all, use_successes=False)
            plt.title(f'{env_id}', fontsize=16)
            plt.xlabel('Batches', fontsize=16)
            plt.ylabel(r'Sampling Error', fontsize=16)
            plt.xticks(fontsize=12, ticks=[32, 64, 96, 128,])
            plt.yticks(fontsize=12)
            plt.yscale('log')
            plt.tight_layout()
            fig.subplots_adjust(top=0.72)
            ax = fig.axes[0]
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
            # plt.legend()

            subplot_i += 1

        save_dir = f'figures/'
        save_name = f'se_{policy}_one'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')

        plt.show()

