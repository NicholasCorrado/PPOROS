import itertools
import os
import warnings
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


def get_data(results_dir, field_name='returns', filename='evaluations.npz', x='updates'):

    try:
        paths = []
        for subdir in sorted(os.listdir(results_dir)):
            if 'run_' in subdir:
                paths.append(f'{results_dir}/{subdir}/{filename}')
    except:
        warnings.warn(f'Data not found at path {results_dir}')
        paths = []

    timesteps = None
    results = []
    first_len = None
    for path in paths:
        with np.load(path) as data:

            vals = data[field_name]
            avg_vals = vals

            if first_len is None:
                first_len = len(avg_vals)
            if len(avg_vals) == first_len:
                # if len(avg_vals) < 25 : continue
                results.append(avg_vals)
                timesteps = data[x]

    return timesteps, np.array(results)


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

    for env_id in ['GridWorld-5x5-v0']:
        for stat in ['se', 'grad_accuracy']:
            results_dict_true = {}
            results_dict_empirical = {}

            i += 1
            color_i = 0
            for algo in ['oracle_adaptive', 'reinforce_props', 'reinforce_on_policy']:
                color_palette = sns.color_palette('colorblind')
                # if b == 1 and s == 512: continue
                results_dir = f"reinforce_fixed3/{env_id}/{algo}"
                timesteps, grad_true = get_data(results_dir=results_dir, field_name='grad_true', x='timesteps')
                timesteps, grad_empirical = get_data(results_dir=results_dir, field_name='grad_empirical', x='timesteps')

                # A warning will be raised when we fail to load from `results_dir`. Skip these failures.
                if len(grad_empirical) > 0:
                    if 'props' in algo:
                        key = "PROPS"
                    if 'on_policy' in algo:
                        key = "On-Policy Sampling"
                    if 'oracle' in algo:
                        key = "Oracle Adaptive Sampling"

                    print(grad_empirical.shape, timesteps.shape)
                    results_dict_empirical[key] = grad_empirical
                    timesteps_dict[key] = timesteps[-len(grad_empirical[0]):]
                    color_dict[key] = color_palette[color_i]
                    color_i += 1

            results_dict_empirical = {algorithm: score for algorithm, score in results_dict_empirical.items()}
            results_dict_true = {algorithm: score for algorithm, score in results_dict_true.items()}

            grad_true = np.mean(grad_true[0, :], axis=0) #[:, None]
            # grad_true = np.tile(grad_true, 100)
            for k, v in results_dict_empirical.items():
                grad_empirical_avg = np.mean(v, axis=0)
                grad_empirical_norm = np.linalg.norm(grad_empirical_avg, axis=-1)
                grad_true_norm = np.linalg.norm(grad_true, axis=-1)

                results_dict[k] = np.array([grad_empirical_avg @ grad_true / (grad_empirical_norm * grad_true_norm)])
                # results_dict[k] = np.array([grad_empirical_avg - grad_true / (grad_empirical_norm * grad_true_norm)])

                # results_dict_empirical[k] = np.mean(v, axis=0)
            # for k, v in results_dict_true.items():
            #     results_dict_true[k] = np.mean(v, axis=0)

            iqm = lambda scores: np.array([AGGR_FUNCS[metric_name]([scores[..., frame]])
                                           for frame in range(scores.shape[-1])])
            iqm_scores, iqm_cis = rly.get_interval_estimates(results_dict, iqm, reps=50)


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
    save_name = f'sampling_error.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=100)
    plt.show()
