import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths, load_data, get_line_styles


def plot(save_dict, name, m=100000, success_threshold=None):
    i = 0

    for agent, info in save_dict.items():
        paths = info['paths']
        x_scale = info['x_scale']
        max_t = info['max_t']
        avgs = []
        for path in paths:
            u, t, avg = load_data(path, name=name, success_threshold=success_threshold)
            if avg is not None:
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

            avg_of_avgs = np.average(avgs, axis=0)

            std = np.std(avgs, axis=0)
            N = len(avgs)
            ci = 1 * std / np.sqrt(N)
            q05 = avg_of_avgs + ci
            q95 = avg_of_avgs - ci

        style_kwargs = get_line_styles(agent)
        style_kwargs['linewidth'] = 2


        x = t_good * x_scale

        if t is None:
            x = np.arange(len(avg_of_avgs))

        style_kwargs['linewidth'] = 3

        if agent == 'PROPS':
            style_kwargs['linestyle'] = '-'
            style_kwargs['linewidth'] = 6
        elif agent == 'PPO-Buffer':
            style_kwargs['linestyle'] = '--'
        elif agent == 'PPO':
            style_kwargs['linestyle'] = ':'
        elif agent == 'PPO-Privileged':
            style_kwargs['linestyle'] = '-.'


        if m:
            x = x[:m]
            avg_of_avgs = avg_of_avgs[:m]
            q05 = q05[:m]
            q95 = q95[:m]

        l = len(avg_of_avgs)
        plt.plot(x[:l], avg_of_avgs, label=agent, **style_kwargs)
        if style_kwargs['linestyle'] == 'None':
            plt.fill_between(x[:l], q05, q95, alpha=0)
        else:
            plt.fill_between(x[:l], q05, q95, alpha=0.2)

        i += 1

if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')
    env_ids = ['Swimmer-v4', 'Hopper-v4','HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4']

    fig = plt.figure(figsize=(6 * 5, 6))
    subplot_i = 1
    for env_id in env_ids:
        plt.subplot(1, 6, subplot_i)
        subplot_i += 1

        for num_steps in [1024,2048,4096,8192]:
            for ros_num_steps in [256, 512, 1024, 2048, 4096]:
                for lr in [1e-3, 1e-4]:
                    for lr_ros in [1e-3,1e-4,1e-5]:
                        for ros_lambda in [0.01, 0.1, 0.3]:
                            for kl in [0.03, 0.05, 0.1]:
                                for mb in [16,32]:
                                    root_dir = f'./data/se_rl/results/{env_id}'
                                    results_dir = f'{root_dir}/ppo_ros/b_2/s_{num_steps}/s_{ros_num_steps}/lr_{lr}/lr_{lr_ros}/kl_0.03/kl_{kl}/l_{ros_lambda}/e_16/mb_{mb}/c_0.3/a_0'

                                    path_dict_all = {}
                                    path_dict_all_ref = {}

                                    ### PROPS SAMPLING ERROR ###########################################################
                                    key = rf'PROPS'
                                    path_dict_aug = get_paths(
                                        results_dir=results_dir,
                                        key=key,
                                        evaluations_name='stats')
                                    path_dict_all.update(path_dict_aug)
                                    if len(path_dict_aug[key]['paths']) == 0:
                                        continue

                                    ### ON-POLICY SAMPLING ERROR #######################################################
                                    key = rf'OS'
                                    path_dict_aug = get_paths(
                                        results_dir=results_dir,
                                        key=key,
                                        evaluations_name='stats')
                                    path_dict_all_ref.update(path_dict_aug)
                                    if len(path_dict_aug[key]['paths']) == 0:
                                        continue

                                    name = 'kl_mle_target'
                                    name_ref = 'ref_kl_mle_target'

                                    plot(path_dict_all, name=name) # PROPS SAMPLING ERROR
                                    plot(path_dict_all_ref, name=name_ref) # ON-POLICY SAMPLING ERROR

                                    plt.title(f'{env_id}', fontsize=24)
                                    plt.xlabel('Timestep', fontsize=24)
                                    plt.ylabel('Sampling Error', fontsize=24)
                                    plt.xticks(fontsize=24)
                                    plt.yticks(fontsize=24)
                                    ax = fig.axes[0]
                                    ax.xaxis.get_offset_text().set_fontsize(24)
                                    plt.tight_layout()

    fig.subplots_adjust(top=0.7)
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=36)


    save_dir = f'figures'
    save_name = f'se_rl.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
