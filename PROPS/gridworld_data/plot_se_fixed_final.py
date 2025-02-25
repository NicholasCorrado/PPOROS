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

    nrows = 2
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))
    if nrows > 1 or ncols > 1:
        axs = axs.flatten()
    else:
        axs = np.array([axs])

    x = 'timesteps'
    b = 1
    i = -1
    # r, c = -1, -1
    env_id = 'GridWorld-10x10-v0'

    for env_id in ['GridWorld-5x5-v0',]:
        for stat in ['se', 'grad_accuracy']:
            results_dict = {}
            i += 1
            color_i = 0
            for algo in ['oracle_adaptive', 'reinforce_props', 'reinforce_on_policy']:
                color_palette = sns.color_palette('colorblind')
                # if b == 1 and s == 512: continue
                results_dir = f"se_fixed_final/{env_id}/{algo}"
                timesteps, results = get_data(results_dir=results_dir, field_name=stat, x='timesteps')

                # A warning will be raised when we fail to load from `results_dir`. Skip these failures.
                if len(results) > 0:
                    if 'props' in algo:
                        key = "PROPS"
                    if 'on_policy' in algo:
                        key = "On-Policy Sampling"
                    if 'oracle' in algo:
                        key = "Oracle Adaptive Sampling"

                    print(results.shape, timesteps.shape)
                    results_dict[key] = results
                    timesteps_dict[key] = timesteps[-len(results[0]):]
                    color_dict[key] = color_palette[color_i]
                    color_i += 1
            #
            # algos = [,]
            # for algo in algos:
            #     # if b == 1 and s == 512: continue
            #     for plr in [0.05]:
            #         results_dir = f"reinforce_fixed2/{env_id}/{algo}/"
            #         timesteps, results = get_data(results_dir=results_dir, field_name=stat, x=x)
            #
            #         # A warning will be raised when we fail to load from `results_dir`. Skip these failures.
            #         if len(results) > 0:
            #             key = f"{algo}"
            #             print(results.shape, timesteps.shape)
            #             results_dict[key] = results
            #             timesteps_dict[key] = timesteps[-len(results[0]):]
            #             color_dict[key] = color_palette[color_i]
            #             color_i += 1

            results_dict = {algorithm: score for algorithm, score in results_dict.items()}
            iqm = lambda scores: np.array([AGGR_FUNCS[metric_name]([scores[..., frame]])
                                           for frame in range(scores.shape[-1])])
            iqm_scores, iqm_cis = rly.get_interval_estimates(results_dict, iqm, reps=100)

            ax = axs[i]
            ax.set_title(f'{env_id}: Fixed Policy')

            ylabel = 'Sampling Error (TV Distance)' if stat == 'se' else r'Gradient Accuracy $\widehat{\nabla_\theta J(\theta)} \cdot \nabla_\theta J(\theta)$ '

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
            # plt.axhline(y=0.97, color='k', linestyle='--')

            # ax.set_ylim(*YLIMS[env_id])

            # plt.ylim(1e-4, 1.7)
            ax.set_xscale('log')
            if stat == 'se':
                ax.set_yscale('log')

    # plt.suptitle('Training Curves')
    plt.tight_layout()

    # # Push plots down to make room for the the legend
    fig.subplots_adjust(top=0.90)
    # # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize='large')

    # ax.legend(ncol=1, fontsize='large')


    save_dir = f'figures'
    save_name = f'sampling_error_final.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=100)
    plt.show()
