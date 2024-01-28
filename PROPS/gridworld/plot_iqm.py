import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths, plot, load_data

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from rliable.plot_utils import _decorate_axis, _annotate_and_decorate_axis, plot_sample_efficiency_curve


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

def get_data(path_dict):
  results = {}
  timesteps = {}
  for agent, info in path_dict.items():
    paths = info['paths']

    t, avgs = load_data(paths, name='returns')
    results[agent] = avgs

  return results

def get_data(path_dict):
  returns = {}
  timesteps = {}
  for agent, info in path_dict.items():
    paths = info['paths']

    t, avgs = load_data(paths, name='returns')
    returns[agent] = avgs
    timesteps[agent] = t

  return returns, timesteps

if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')
    env_ids = ['TwoStep-v0']

    fig = plt.figure(figsize=(1*6,1*6))
    i = 1
    for env_id in env_ids:
        ax = plt.subplot(1, 1, i)
        i+=1

        path_dict_all = {}
        ### PROPS ##################################################################################################
        key = rf'PPO with Adaptive Sampling (PROPS)'
        algo = 'ppo_props'
        results_dir = f'results/{env_id}/{algo}/b_1'
        path_dict_aug = get_paths(
            results_dir=results_dir,
            key=key,
            evaluations_name='evaluations')
        if len(path_dict_aug[key]['paths']) > 0:
            path_dict_all.update(path_dict_aug)

        ### PPO #########################################################################################
        key = rf'PPO with On-Policy Sampling'
        algo = 'ppo_buffer'
        results_dir = f'results/{env_id}/{algo}/b_1'
        path_dict_aug = get_paths(
            results_dir=results_dir,
            key=key,
            evaluations_name='evaluations')
        if len(path_dict_aug[key]['paths']) > 0:
            path_dict_all.update(path_dict_aug)


        return_dict, timestep_dict = get_data(path_dict_all)

        algorithms = list(return_dict.keys())
        ale_frames_scores_dict = {algorithm: score for algorithm, score
                                  in return_dict.items()}
        iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                                       for frame in range(scores.shape[-1])])
        iqm_scores, iqm_cis = rly.get_interval_estimates(
            ale_frames_scores_dict, iqm, reps=2000)
        plot_sample_efficiency_curve(
            timestep_dict, iqm_scores, iqm_cis, algorithms=algorithms,
            xlabel=r'Timestep',
            ylabel='IQM Return')
        # plt.title(f'{env_id}', fontsize='xx-large')
        #
        # # Use scientific notation for x-axis
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # set fontsize of 1e6
        ax.xaxis.get_offset_text().set_fontsize('xx-large')
        #
        plt.tight_layout()
        # plt.show()

        #
        # plot(path_dict_all, name='returns')
        # # plt.title(f'', fontsize=20)
        # plt.xlabel('Timestep', fontsize=20)
        # plt.ylabel('Return', fontsize=20)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # # plt.ylim(0.7,1.05)
        # # plt.xscale('log')
        # # Use scientific notation for x-axis
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # # set fontsize of 1e6
        # ax.xaxis.get_offset_text().set_fontsize(14)
        #
        # plt.tight_layout()
        #
        # # Push plots down to make room for the the legend
        # fig.subplots_adjust(top=0.8)
        #
        # # Fetch and plot the legend from one of the subplots.
        # ax = fig.axes[0]
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center', ncol=1, fontsize=17)
        #
        # save_dir = f'figures'
        # save_name = f'discrete_return.png'
        # os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(f'{save_dir}/{save_name}')
        #
        # plt.show()


        data_dict = get_data(path_dict_all)
        for k, v in data_dict.items():
            print(k, v.shape[0])
        thresholds = np.linspace(0.5, 1.999, 101)

        algorithms = list(data_dict.keys())

        score_distributions, score_distributions_cis = rly.create_performance_profile(
            data_dict, thresholds, reps=100)
        # Plot score distributions
        fig, ax = plt.subplots(ncols=1, figsize=(6, 6))
        plot_utils.plot_performance_profiles(
            score_distributions, thresholds,
            performance_profile_cis=score_distributions_cis,
            colors=dict(zip(algorithms, seaborn.color_palette('colorblind'))),
            xlabel=r'Return $(\tau)$',
            # linestyles=linestyles,
            # alpha=0,
            ax=ax)
        plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5)
        plt.ylabel('Fraction of runs ' r'with score > $\tau$', fontsize='xx-large')
        plt.xlabel(r'Return $(\tau)$', fontsize='xx-large')

        plt.tight_layout()
        plt.show()
