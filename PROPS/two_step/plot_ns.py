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

            avgs.append(avg[[-1]])

    return t, np.array(avgs)

def get_data(path_dict):
  results = {}
  timesteps = {}
  for agent, info in path_dict.items():
    paths = info['paths']

    t, avgs = load_data(paths, name='returns')
    results[agent] = avgs

  return results

if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid', palette='colorblind')

    env_ids = ['TwoStep-v0']


    root_dir = 'condor/results'
    root_dir = 'ns1/results'

    for ns in [2, 10, 20, 25, 50]:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(1 * 6 * 2, 1 * 4))
        i = 1

        for env_id in env_ids:
            ax = plt.subplot(1, 2, i)
            i+=1

            path_dict_all = {}
            ### PROPS ##################################################################################################
            key = rf'PROPS, $b=1$'
            algo = 'ppo_props'
            results_dir = f'{root_dir}/{env_id}/{algo}/b_1/ns_{ns}'
            path_dict_aug = get_paths(
                results_dir=results_dir,
                key=key,
                evaluations_name='evaluations')
            if len(path_dict_aug[key]['paths']) > 0:
                path_dict_all.update(path_dict_aug)

            ### PROPS ##################################################################################################
            key = rf'PROPS, $b=2$'
            algo = 'ppo_props'
            results_dir = f'{root_dir}/{env_id}/{algo}/b_2/ns_{ns}'
            path_dict_aug = get_paths(
                results_dir=results_dir,
                key=key,
                evaluations_name='evaluations')
            if len(path_dict_aug[key]['paths']) > 0:
                path_dict_all.update(path_dict_aug)

            ### PPO #########################################################################################
            key = rf'PPO, $b=1$'
            algo = 'ppo_buffer'
            results_dir = f'{root_dir}/{env_id}/{algo}/b_1/ns_{ns}'
            path_dict_aug = get_paths(
                results_dir=results_dir,
                key=key,
                evaluations_name='evaluations')
            if len(path_dict_aug[key]['paths']) > 0:
                path_dict_all.update(path_dict_aug)

            key = rf'PPO, $b=2$'
            algo = 'ppo_buffer'
            results_dir = f'{root_dir}/{env_id}/{algo}/b_2/ns_{ns}'
            path_dict_aug = get_paths(
                results_dir=results_dir,
                key=key,
                evaluations_name='evaluations')
            if len(path_dict_aug[key]['paths']) > 0:
                path_dict_all.update(path_dict_aug)

            plot(path_dict_all, name='returns')
            plt.title(f'{ns}', fontsize=20)
            plt.xlabel('Timestep', fontsize=20)
            plt.ylabel('Average Return', fontsize=20)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(alpha=0.2)
            # plt.ylim(0.7,1.05)
            # plt.xscale('log')
            # Use scientific notation for x-axis
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            # set fontsize of 1e6
            ax.xaxis.get_offset_text().set_fontsize(14)

            plt.tight_layout()


            save_dir = f'figures'
            save_name = f'discrete_return.png'
            os.makedirs(save_dir, exist_ok=True)
            # plt.savefig(f'{save_dir}/{save_name}')
            # plt.savefig(f'2step_return', dpi=300)
            # plt.show()


            data_dict = get_data(path_dict_all)
            for k, v in data_dict.items():
                print(k, v.shape[0])
            thresholds = np.linspace(0.5, 1.999, 101)

            algorithms = list(data_dict.keys())

            score_distributions, score_distributions_cis = rly.create_performance_profile(
                data_dict, thresholds, reps=500)
            # Plot score distributions
            ax = plt.subplot(1, 2, 2)
            plot_utils.plot_performance_profiles(
                score_distributions, thresholds,
                performance_profile_cis=score_distributions_cis,
                colors=dict(zip(algorithms, seaborn.color_palette('colorblind'))),
                xlabel=r'Return $(\tau)$',
                # linestyles=linestyles,
                # alpha=0,
                ax=ax)
            plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5)
            plt.ylabel('Fraction of runs\n' r'with score > $\tau$', fontsize='xx-large')
            plt.xlabel(r'Return $(\tau)$', fontsize='xx-large')

            plt.tight_layout()

            # Push plots down to make room for the the legend
            fig.subplots_adjust(top=0.8)

            # Fetch and plot the legend from one of the subplots.
            ax = fig.axes[0]
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=20)

            plt.savefig(f'2step_results', dpi=300)

            plt.show()
