import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths, load_data, plot, get_line_styles

ylims = {
    'Hopper-v4': (0, 4000),
    'HalfCheetah-v4': (0, 4000),
    'Walker2d-v4': (0, 400),
    'Ant-v4': (0, 6000),
    'InvertedPendulum-v4': (0, 1100),
    'InvertedDoublePendulum-v4': (0, 10000),
    'Acrobot-v1': (-500, 0),
    'CartPole-v1': (0, 600),
    'LunarLander-v2': (-250, 300),
    'Swimmer-v4': (0, 150),
    'Humanoid-v4': (0, 6500)
}

if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid', palette='colorblind')
    env_ids = ['Humanoid-v4']




    fig = plt.figure(figsize=(3, 3))
    subplot_i = 0
    for env_id in env_ids:
        subplot_i += 1
        plt.subplot(1, 1, subplot_i)

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
                                    key = rf'On-Policy Sampling'
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

                                    plt.title(f'{env_id}', fontsize=16)
                                    plt.xlabel('Timestep', fontsize=16)
                                    plt.ylabel('Sampling Error', fontsize=16)
                                    plt.xticks(fontsize=12)
                                    plt.yticks(fontsize=12)
                                    ax = fig.axes[0]
                                    ax.xaxis.get_offset_text().set_fontsize(12)
                                    plt.tight_layout()

    fig.subplots_adjust(top=0.7)
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=1, fontsize=12)


    save_dir = f'figures'
    save_name = f'se_rl_one.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
