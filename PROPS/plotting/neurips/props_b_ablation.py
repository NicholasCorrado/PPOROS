import os

import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths, plot

if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')
    env_ids = ['Swimmer-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4']

    fig = plt.figure(figsize=(3*3,2*3))
    i = 1
    for env_id in env_ids:
        ax = plt.subplot(2, 3, i)
        i+=1
        for b in [2,3,4,8]:

            path_dict_all = {}

            algo = 'ppo_ros'
            for num_steps in [1024, 2048, 4096, 8192]:
                for ros_num_steps in [256, 512, 1024, 2048, 4096]:
                    for lr in [1e-3, 1e-4]:
                        for lr_ros in [1e-3, 1e-4, 1e-5]:
                            for ros_lambda in [0, 0.01, 0.1, 0.3]:
                                for kl in [0.01, 0.03, 0.05, 0.1]:
                                    for mb in [8, 16, 32]:
                                        root_dir = f'data/props_b_ablation/results/{env_id}'
                                        results_dir = f'{root_dir}/{algo}/b_{b}/' \
                                                      f's_{num_steps}/s_{ros_num_steps}/' \
                                                      f'lr_{lr}/lr_{lr_ros}/' \
                                                      f'kl_0.03/kl_{kl}/' \
                                                      f'l_{ros_lambda}/' \
                                                      f'e_16/mb_{mb}/c_0.1/a_0'
                                        key = rf'$b = {b}$'
                                        path_dict_aug = get_paths(
                                            results_dir=results_dir,
                                            key=key,
                                            evaluations_name='evaluations')
                                        if len(path_dict_aug[key]['paths']) == 0: continue
                                        path_dict_all.update(path_dict_aug)

            plot(path_dict_all, name='returns')
            plt.title(f'{env_id}', fontsize=20)
            plt.xlabel('Timestep', fontsize=20)
            plt.ylabel('Return', fontsize=20)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_fontsize(14)
            plt.tight_layout()

    fig.subplots_adjust(top=0.85)
    ax = fig.axes[1]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=17)

    save_dir = f'figures'
    save_name = f'props_b_ablation.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}')

    plt.show()
