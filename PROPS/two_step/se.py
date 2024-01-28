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

            # if avg[-1] > 1.9:
            avgs.append(avg)

    return t, np.array(avgs)

if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid', palette='colorblind')
    env_ids = ['TwoStep-v0']




    fig = plt.figure(figsize=(3, 3))
    subplot_i = 0
    for env_id in env_ids:
        subplot_i += 1
        plt.subplot(1, 1, subplot_i)

        root_dir = f'./results/TwoStep-v0/'

        path_dict_all = {}
        path_dict_all_ref = {}

        ### PROPS SAMPLING ERROR ###########################################################
        # key = rf'PROPS'
        # path_dict_aug = get_paths(
        #     results_dir=results_dir,
        #     key=key,
        #     evaluations_name='stats')
        # path_dict_all.update(path_dict_aug)
        # if len(path_dict_aug[key]['paths']) == 0:
        #     continue

        ### ON-POLICY SAMPLING ERROR #######################################################
        key = rf'On-Policy Sampling'
        results_dir = f'{root_dir}/ppo_buffer/b_1'
        path_dict_aug = get_paths(
            results_dir=results_dir,
            key=key,
            evaluations_name='stats')
        path_dict_all.update(path_dict_aug)
        if len(path_dict_aug[key]['paths']) == 0:
            continue

        for key, paths in path_dict_all.items():
            data = load_data(paths, name='traj_probs_error')

            plot(path_dict_all, name='traj_probs_error') # PROPS SAMPLING ERROR

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
