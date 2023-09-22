import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths


def load_data(path, successes=False, success_threshold=None):
    with np.load(path, allow_pickle=False) as data:
        # u = data['updates']
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
            ci = std / np.sqrt(N) * 1.96
            q05 = avg_of_avgs - ci
            q95 = avg_of_avgs + ci

        style_kwargs = {'linewidth': 3}
        if 'PPO' in agent:
            style_kwargs['color'] = 'k'
            style_kwargs['linestyle'] = ':'
            style_kwargs = {'linewidth': 3}

        if 'PROPS' in agent:
            style_kwargs['color'] = 'k'
            style_kwargs['linestyle'] = ':'
            style_kwargs = {'linewidth': 6}

        if 'Buffer' in agent:
            style_kwargs['linestyle'] = '--'

        plt.plot(t, avg_of_avgs, label=agent, **style_kwargs)
        plt.fill_between(t, q05, q95, alpha=0.2)

        i += 1


if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')
    env_ids = ['Swimmer-v4', 'Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4']

    for policy in ['expert', 'random']:
        subplot_i = 1
        fig = plt.figure(figsize=(6 * 5, 6))

        for env_id in env_ids:
            plt.subplot(1, 6, subplot_i)
            path_dict_all = {}

            root_dir = f'data/se_fixed/results/{env_id}'

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
            plt.title(f'{env_id}', fontsize=32)
            plt.xlabel('Batches', fontsize=32)
            plt.ylabel(r'Sampling Error', fontsize=32)
            plt.xticks(fontsize=32, ticks=[32, 64, 96, 128,])
            plt.yticks(fontsize=32)
            plt.yscale('log')
            plt.tight_layout()

            subplot_i += 1

        fig.subplots_adjust(top=0.7)
        ax = fig.axes[0]
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=36)

        save_dir = f'figures'
        save_name = f'se_fixed_{policy}'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}')

        plt.show()

