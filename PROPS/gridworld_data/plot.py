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
    axs = np.array([axs])
    # axs = axs.flatten()

    x = 'updates'
    b = 1
    i = -1
    # r, c = -1, -1
    env_id = 'GridWorld-5x5-v0'
    for stat in ['returns']:
        i += 1
        results_dict = {}

        algos = ['actor_critic_on_policy']
        for algo in algos:
            color_palette = sns.color_palette('colorblind')
            color_i = 0
            for s in [10, 40, 80, 160]:
            # for s in [20]:
                for b in [1,]:
                    for lr in [1e-3]:
                        # if b == 1 and s == 512: continue
                        results_dir = f"condor/gw5/results/{env_id}/{algo}/" \
                                      f"lr_{lr}/s_{s}/b_{b}"
                        timesteps, results = get_data(results_dir=results_dir, field_name=stat, x=x)

                        # A warning will be raised when we fail to load from `results_dir`. Skip these failures.
                        if len(results) > 0:
                            key = f"s={s}, b={b}"
                            if s == 1024:
                                key = f"{algo} privileged"
                                timesteps = timesteps/2
                            # key = f"{s}"
                            print(results.shape, timesteps.shape)
                            # if stat == 'se':
                                # results /= 32
                            results_dict[key] = results
                            timesteps_dict[key] = timesteps[-len(results[0]):]
                            color_dict[key] = color_palette[color_i]
                            color_i += 1
                            if s==50:
                                linestyles_dict[key] = ':'
                            else:
                                linestyles_dict[key] = '-'

            color_i = 1
            s = 10
            for b in [4, 8, 16]:
                for lr in [1e-3]:
                    # if b == 1 and s == 512: continue
                    results_dir = f"condor/gw5/results/{env_id}/{algo}/" \
                                  f"lr_{lr}/s_{s}/b_{b}"
                    timesteps, results = get_data(results_dir=results_dir, field_name=stat)

                    # A warning will be raised when we fail to load from `results_dir`. Skip these failures.
                    if len(results) > 0:
                        key = f"s={s}, b={b}"
                        print(results.shape, timesteps.shape)
                        results_dict[key] = results
                        timesteps_dict[key] = timesteps[-len(results[0]):]
                        color_dict[key] = color_palette[color_i]
                        color_i += 1
                        linestyles_dict[key] = ':'

        results_dict = {algorithm: score for algorithm, score in results_dict.items()}
        iqm = lambda scores: np.array([AGGR_FUNCS[metric_name]([scores[..., frame]])
                                       for frame in range(scores.shape[-1])])
        iqm_scores, iqm_cis = rly.get_interval_estimates(results_dict, iqm, reps=500)

        ax = axs[i]
        ax.set_title(f'{env_id}')

        ylabel = f'{metric_name} Return'.title() if stat == 'returns' else 'Sampling Error (TV Distance)'

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


    # plt.suptitle('Training Curves')
    plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.60)
    # # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize='large')

    save_dir = f'figures'
    save_name = f'gridworld.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=100)
    plt.show()
