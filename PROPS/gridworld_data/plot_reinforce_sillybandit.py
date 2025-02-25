import itertools
import os
from collections import defaultdict

import numpy as np

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from PROPS.plotting_new.utils import get_data, plot_sample_efficiency_curve

from rliable import library as rly
from rliable import metrics

if __name__ == "__main__":


    timesteps_dict = {}
    results_dict = {}
    linestyles_dict = {}
    color_dict = {}

    metric_name = 'mean'
    AGGR_FUNCS = {
        "mean": metrics.aggregate_mean,
        "iqm": metrics.aggregate_iqm,
    }
    env_ids = [
        'GridWorld-5x5-v0',
    ]

    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
    if nrows > 1 or ncols > 1:
        axs = axs.flatten()
    else:
        axs = np.array([axs])

    x = 'updates'
    b = 1
    i = -1
    # r, c = -1, -1
    stat = 'ppo_entropy_loss'

    env_ids = ['SillyBandit-v0']

    results_dict = {}
    i += 1
    for env_id in env_ids:
        color_i = 0

        algos = ['reinforce_on_policy',]
        for algo in algos:
            color_palette = sns.color_palette('colorblind')
            for lr in [1e-1,]:
                for s in [1, 5, 10, 50, 100]:
                    results_dir = f"reinforce_sillybandit/{env_id}/{algo}/lr_{lr}/s_{s}"
                    timesteps, results = get_data(results_dir=results_dir, field_name=stat, x=x)

                    # A warning will be raised when we fail to load from `results_dir`. Skip these failures.
                    if len(results) > 0:
                        key = f"batch size = {s}"
                        print(results.shape, timesteps.shape)
                        results_dict[key] = results
                        timesteps_dict[key] = timesteps[-len(results[0]):]
                        color_dict[key] = color_palette[color_i]
                        color_i += 1




    results_dict = {algorithm: score for algorithm, score in results_dict.items()}
    iqm = lambda scores: np.array([AGGR_FUNCS[metric_name]([scores[..., frame]])
                                   for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(results_dict, iqm, reps=5000)

    ax = axs[i]
    ax.set_title(f'10-arm Bandit')

    ylabel = f'{metric_name} Return'.title() if stat == 'returns' else 'Policy Entropy'

    plot_sample_efficiency_curve(
        timesteps_dict, # assumes `timesteps` is the same for all curves
        iqm_scores,
        iqm_cis,
        ax=ax,
        colors=color_dict,
        algorithms=None,
        # marker=None,
        linestyles=linestyles_dict,
        xlabel=x.capitalize(),
        ylabel=ylabel,
        labelsize=12,
        ticklabelsize=12,
        # legend=True,
    )

        # ax.set_ylim(*YLIMS[env_id])

    # plt.ylim(1e-4, 1.7)
    # plt.yscale('log')
    # plt.xscale('log')


    # plt.suptitle('Training Curves')
    plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.65)
    # # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=1, fontsize='large')

    # ax.legend(ncol=1, fontsize='large')


    save_dir = f'figures'
    save_name = f'sillybandit.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=100)
    plt.show()
