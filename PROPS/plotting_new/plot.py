import itertools
import os

import numpy as np

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn

from PROPS.plotting.utils import get_data, plot_sample_efficiency_curve

from rliable import library as rly
from rliable import metrics


TIMESTEPS = {
    'Hopper-v4': int(10e6),
    'HalfCheetah-v4': int(10e6),
    'Ant-v4': int(10e6),
    'Walker2d-v4': int(10e6),
    'Humanoid-v4': int(10e6),
    'Swimmer-v4': int(10e6),
    'InvertedPendulum-v4': int(200e3),
    'InvertedDoublePendulum-v4': int(10e6),
    'Pendulum-v1': int(5e6),
    'Reacher-v4': int(5e6),
}

YLIMS = {
    'Hopper-v4': (0, 3500),
    'HalfCheetah-v4': (0, 4000),
    'Ant-v4': (0, 6000),
    'Walker2d-v4': (0, 5000),
    'Humanoid-v4': (0, 6000),
    'Swimmer-v4': (-50, 200),
    'InvertedPendulum-v4': (0, 1200),
    'InvertedDoublePendulum-v4': (0, 11000),
    'Pendulum-v1': (-1200, -100),
    'Reacher-v4': (-70, 0),
}

if __name__ == "__main__":


    timesteps_dict = {}
    results_dict = {}



    algos = ['qmix']

    metric_name = 'mean'
    AGGR_FUNCS = {
        "mean": metrics.aggregate_mean,
        "iqm": metrics.aggregate_iqm,
    }
    hparams = [
        [1,],
        [1e-3, 1e-4, 1e-5],
        [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    ]
    env_ids = ['Pendulum-v1', 'InvertedPendulum-v4', 'InvertedDoublePendulum-v4', 'Reacher-v4',
               'Swimmer-v4', 'HalfCheetah-v4', 'Hopper-v4']

    nrows = len(env_ids)
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*6))

    r, c = -1, -1
    for env_id in env_ids:
        # add all results you want to plot on a single subplot
        r += 1
        for algo in ['reinforce']:
            for sampling in ['on_policy']:
                c = -1
                for lr in [1e-3, 1e-4, 1e-5]:
                    c += 1
                    results_dict = {}

                    for s in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                        b = 1
                # for b, lr, s, in itertools.product(*hparams):
                        results_dir = f"../../chtc/results/reinforce/results/{env_id}/{algo}_{sampling}" \
                                      f"/b_{b}/s_{s}/lr_{lr}"
                        timesteps, results = get_data(results_dir=results_dir)

                        # A warning will be raised when we fail to load from `results_dir`. Skip these failures.
                        if len(results) > 0:
                            mask = timesteps < TIMESTEPS[env_id]
                            timesteps = timesteps[mask]
                            results = results[:, mask]

                            key = f"/b_{b}/s_{s}/lr_{lr}"
                            key = f"{s}"
                            results_dict[key] = results
                            timesteps_dict[key] = timesteps

                    results_dict = {algorithm: score for algorithm, score in results_dict.items()}
                    iqm = lambda scores: np.array([AGGR_FUNCS[metric_name]([scores[..., frame]])
                                                   for frame in range(scores.shape[-1])])
                    iqm_scores, iqm_cis = rly.get_interval_estimates(results_dict, iqm, reps=5)

                    ax = axs[r, c]
                    ax.set_title(f'{env_id}')

                    plot_sample_efficiency_curve(
                        timesteps_dict, # assumes `timesteps` is the same for all curves
                        iqm_scores,
                        iqm_cis,
                        ax=ax,
                        algorithms=None,
                        # marker=None,
                        xlabel='Timesteps',
                        ylabel=f'{metric_name} Return'.title(),
                        labelsize=12,
                        ticklabelsize=12,
                        legend=True,
                    )
                    ax.set_ylim(*YLIMS[env_id])


    # plt.suptitle('Training Curves')
    plt.tight_layout()

    # Push plots down to make room for the the legend
    # fig.subplots_adjust(left=0.1, top=0.75)
    # # Fetch and plot the legend from one of the subplots.
    # ax = fig.axes[0]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='large')

    save_dir = f'figures'
    save_name = f'return_{metric_name}.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=300)
    plt.show()
