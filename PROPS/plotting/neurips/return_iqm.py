import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths, plot, load_data

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from rliable.plot_utils import _decorate_axis



def plot_interval_estimates(point_estimates,
                            interval_estimates,
                            metric_names,
                            algorithms=None,
                            colors=None,
                            color_palette='colorblind',
                            max_ticks=4,
                            ax=None,
                            subfigure_width=3.4,
                            row_height=0.37,
                            xlabel_y_coordinate=-0.1,
                            xlabel='Normalized Score',
                            **kwargs):
  """Plots various metrics with confidence intervals.

  Args:
    point_estimates: Dictionary mapping algorithm to a list or array of point
      estimates of the metrics to plot.
    interval_estimates: Dictionary mapping algorithms to interval estimates
      corresponding to the `point_estimates`. Typically, consists of stratified
      bootstrap CIs.
    metric_names: Names of the metrics corresponding to `point_estimates`.
    algorithms: List of methods used for plotting. If None, defaults to all the
      keys in `point_estimates`.
    colors: Maps each method to a color. If None, then this mapping is created
      based on `color_palette`.
    color_palette: `seaborn.color_palette` object for mapping each method to a
      color.
    max_ticks: Find nice tick locations with no more than `max_ticks`. Passed to
      `plt.MaxNLocator`.
    subfigure_width: Width of each subfigure.
    row_height: Height of each row in a subfigure.
    xlabel_y_coordinate: y-coordinate of the x-axis label.
    xlabel: Label for the x-axis.
    **kwargs: Arbitrary keyword arguments.

  Returns:
    fig: A matplotlib Figure.
    axes: `axes.Axes` or array of Axes.
  """

  if algorithms is None:
    algorithms = point_estimates.keys()
  try:
    num_metrics = len(point_estimates[algorithms[0]])
  except:
      return None, None
  # figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
  # fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
  if colors is None:
    color_palette = seaborn.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))
  h = kwargs.pop('interval_height', 0.6)

  # for idx, metric_name in enumerate(metric_names):
  idx = 0
  metric_name = 'IQM'
  for alg_idx, algorithm in enumerate(algorithms):
    ax = ax
      # Plot interval estimates.
    lower, upper = interval_estimates[algorithm][:, idx]
    ax.barh(
          y=alg_idx,
          width=upper - lower,
          height=h,
          left=lower,
          color=colors[algorithm],
          alpha=0.75,
          label=algorithm)
    # Plot point estimates.
    ax.vlines(
          x=point_estimates[algorithm][idx],
          ymin=alg_idx - (7.5 * h / 16),
          ymax=alg_idx + (6 * h / 16),
          label=algorithm,
          color='k',
          alpha=0.5)

    ax.set_yticks(list(range(len(algorithms))))
    ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    # if idx != 0:
    # ax.set_yticks([])
    # else:
    # ax.set_yticklabels(algorithms, fontsize='large')
    # ax.set_title(metric_name, fontsize='large')
    ax.tick_params(axis='both', which='major')
    _decorate_axis(ax, ticklabelsize='large', wrect=5)
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.25)
  fig.text(0.4, xlabel_y_coordinate, xlabel, ha='center', fontsize='large')
  plt.subplots_adjust(wspace=kwargs.pop('wspace', 0.11), left=0.0)
  return fig

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

            avgs.append(avg)

    return t, np.array(avgs)

def get_data(path_dict):
  results = {}
  for agent, info in path_dict.items():
    paths = info['paths']

    t, avgs = load_data(paths, name='returns')
    results[agent] = avgs

  return results

PROPS_PATHS = {
    'Swimmer-v4': {
        2: {
            'path': 'data/rl/props/results/Swimmer-v4/ppo_ros/b_2/s_2048/s_1024/lr_0.001/lr_1e-05/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.3/a_0',
        }
    },
    'HalfCheetah-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/HalfCheetah-v4/ppo_props/b_1/s_1024/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.03/l_0.3/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/HalfCheetah-v4/ppo_ros/b_2/s_1024/s_512/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/l_0.3/e_16/mb_16/c_0.3/a_0',
        }
    },
    'Hopper-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/Hopper-v4/ppo_props/b_1/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.1/l_0.3/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/Hopper-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.05/l_0.3/e_16/mb_16/c_0.3/a_0',
        }
    },
    'Walker2d-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/Walker2d-v4/ppo_props/b_1/s_2048/s_256/lr_0.001/lr_0.0001/kl_0.03/kl_0.05/l_0.1/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/Walker2d-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.1/l_0.3/e_16/mb_32/c_0.3/a_0',
        }
    },
    'Ant-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/Ant-v4/ppo_props/b_1/s_1024/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.03/l_0.1/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/Ant-v4/ppo_ros/b_2/s_2048/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.05/a_0',
        }
    },
    'Humanoid-v4': {
        1: {
            'path': '/Users/nicholascorrado/code/josiah/PPOROS/PROPS/plotting/props_b1/results/Humanoid-v4/ppo_props/b_1/s_8192/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.03/l_0.3/c_0.3',
        },
        2: {
            'path': 'data/rl/props/results/Humanoid-v4/ppo_ros/b_2/s_8192/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.1/l_0.1/e_16/mb_32/c_0.3/a_0',
        },
    },
}

AWPROPS_PATHS = {
    'Swimmer-v4': {
        2: {
            # 'path': 'data/rl/props/results/Swimmer-v4/ppo_ros/b_2/s_2048/s_1024/lr_0.001/lr_1e-05/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.3/a_0',
        }
    },
    'HalfCheetah-v4': {
        2: {
            # 'path': 'data/rl/props/results/HalfCheetah-v4/ppo_ros/b_2/s_1024/s_512/lr_0.0001/lr_0.001/kl_0.03/kl_0.05/l_0.3/e_16/mb_16/c_0.3/a_0',
        }
    },
    'Hopper-v4': {
        2: {
            'path': '/Users/nicholascorrado/code/PROPS/PROPS/plotting/7_04/adv_best/results/Hopper-v4/ppo_props/b_2/s_1024/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.1/l_0.01/c_0.3',
        }
    },
    'Walker2d-v4': {
        2: {
            # 'path': 'data/rl/props/results/Walker2d-v4/ppo_ros/b_2/s_2048/s_256/lr_0.001/lr_0.001/kl_0.03/kl_0.1/l_0.3/e_16/mb_32/c_0.3/a_0',
        }
    },
    'Ant-v4': {
        2: {
            # 'path': 'data/rl/props/results/Ant-v4/ppo_ros/b_2/s_2048/s_256/lr_0.0001/lr_0.001/kl_0.03/kl_0.03/l_0.1/e_16/mb_16/c_0.05/a_0',
        }
    },
    'Humanoid-v4': {
        2: {
            # 'path': 'data/rl/props/results/Humanoid-v4/ppo_ros/b_2/s_8192/s_256/lr_0.0001/lr_0.0001/kl_0.03/kl_0.1/l_0.1/e_16/mb_32/c_0.3/a_0',
        },
    },
}


if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')
    env_ids = ['Swimmer-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4']
    # env_ids = ['Ant-v4']

    fig, axes = plt.subplots(2, 3, figsize=(3*3,2*3))
    axes = axes.flatten()
    i = 0
    for env_id in env_ids:
        i+=1
        ax = plt.subplot(2, 3, i)

        path_dict_all = {}
        print(env_id)
        ### PROPS ##################################################################################################
        key = rf'PROPS'
        algo = 'ppo_ros'
        try:
            results_dir = PROPS_PATHS[env_id][2]['path']
            path_dict_aug = get_paths(
                results_dir=results_dir,
                key=key,
                evaluations_name='evaluations')
            path_dict_all.update(path_dict_aug)
        except:
            pass

        algo = 'ppo_ros'
        ns = [1024, 2048, 4096, 8192]
        lrs = [1e-3, 1e-4]
        ros_lrs = [1e-3, 1e-4, 1e-5]

        for b in [2]:
            for s in ns:
                for lr in lrs:
                    for rlr in ros_lrs:
                        for ros_update_epochs in [16]:
                            for ros_mb in [16, 32]:
                                for l in [0.01, 0.1, 0.3]:
                                    for rs in [256, 512, 1024]:
                                        for rkl in [0.03, 0.05, 0.1]:

                                            results_dir = f'../more/results/{env_id}/{algo}/b_{b}/s_{s}/s_{rs}/lr_{lr}/lr_{rlr}/kl_0.03/kl_{rkl}/l_{l}/e_16/mb_{ros_mb}/c_0.3/a_0'
                                            # results_dir = f'../more_2/results/{env_id}/{algo}/b_{b}/s_{s}/s_{rs}/lr_{lr}/lr_{rlr}/kl_0.03/kl_{rkl}/l_{l}/e_16/mb_{ros_mb}/c_0.3/a_0'
                                            results_dir = f'../props_b1_50/results/{env_id}/{algo}/b_{b}/s_{s}/s_{rs}/lr_{lr}/lr_{rlr}/kl_0.03/kl_{rkl}/l_{l}/e_16/mb_{ros_mb}/c_0.3/a_0'

                                            # key = rf'PROPS: {s},{rs}; {lr},{rlr}; {rkl}; {l}, {ros_mb}'

                                            path_dict_aug = get_paths(
                                                results_dir=results_dir,
                                                key=key,
                                                evaluations_name='evaluations')
                                            path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])

        ### AW-PROPS ##################################################################################################
        # key = rf'AW-PROPS'
        # algo = 'ppo_ros'
        # try:
        #     results_dir = AWPROPS_PATHS[env_id][2]['path']
        # except:
        #     continue
        # path_dict_aug = get_paths(
        #     results_dir=results_dir,
        #     key=key,
        #     evaluations_name='evaluations')
        # path_dict_all.update(path_dict_aug)

        ### PPO-PRIVILEGED #########################################################################################
        key = rf'PPO-Privileged'
        algo = 'ppo_buffer'
        for s in [1024, 2048, 4096, 8192]:
            for lr in [1e-3, 1e-4]:
                for kl in [0.03]:
                    results_dir = f'data/rl/ppo_privileged/results/{env_id}/{algo}/b_1/s_{s}/lr_{lr}/kl_{kl}/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        x_scale=0.5,
                        evaluations_name='evaluations')
                    if len(path_dict_aug[key]['paths']) == 0: continue
                    path_dict_all.update(path_dict_aug)


        ### PPO-BUFFER #############################################################################################
        key = rf'PPO-Buffer'
        algo = 'ppo_buffer'
        for s in [1024, 2048, 4096, 8192]:
            for lr in [1e-3, 1e-4]:
                for kl in [0.03]:
                    results_dir = f'data/rl/ppo_buffer/results/{env_id}/{algo}/b_2/s_{s}/lr_{lr}/kl_0.03/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        evaluations_name='evaluations')
                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all.update(path_dict_aug)

                    results_dir = f'../more_buffer/results/{env_id}/{algo}/b_2/s_{s}/lr_{lr}/kl_0.03/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        evaluations_name='evaluations')
                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])


        ### PPO ####################################################################################################
        key = 'PPO'
        algo = 'ppo_buffer'
        for s in [1024, 2048, 4096, 8192]:
            for lr in [1e-3, 1e-4]:
                for kl in [0.03]:
                    results_dir = f'data/rl/ppo_buffer/results/{env_id}/{algo}/b_1/s_{s}/lr_{lr}/kl_0.03/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        evaluations_name='evaluations')
                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all.update(path_dict_aug)

                    results_dir = f'../more_buffer/results/{env_id}/{algo}/b_1/s_{s}/lr_{lr}/kl_0.03/e_10'

                    path_dict_aug = get_paths(
                        results_dir=results_dir,
                        key=key,
                        evaluations_name='evaluations')
                    if len(path_dict_aug[key]['paths']) > 0:
                        path_dict_all[key]['paths'].extend(path_dict_aug[key]['paths'])


        # plot(path_dict_all, name='returns')
        plt.title(f'{env_id}', fontsize='large')
        # plt.xlabel('Timestep', fontsize='large')
        # plt.ylabel('Return', fontsize=20)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        #
        # # Use scientific notation for x-axis
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # # set fontsize of 1e6
        # ax.xaxis.get_offset_text().set_fontsize(14)


        data_dict = get_data(path_dict_all)

        IQM = lambda x: metrics.aggregate_iqm(
            np.mean(x[:, -5:], axis=-1)
        )
        # IQM = lambda x: metrics.aggregate_iqm(np.max(x,axis=0))
        d = data_dict['PPO'][:, -5:]
        m = d.mean(axis=-1)
        print(data_dict['PPO'][:, -5:].mean(axis=0))
        # IQM = lambda x: metrics.aggregate_mean(x[:, -1:])
        aggregate_func = lambda x: np.array([IQM(x)])
        aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
            data_dict, aggregate_func, reps=50)

        # ax = plt.subplot(nrows, ncols, i)
        ax = axes[i - 1]
        fig = plot_interval_estimates(
            aggregate_scores,
            aggregate_interval_estimates,
            metric_names=[f'{env_id}'],
            algorithms=['PROPS', 'PPO-Privileged', 'PPO-Buffer', 'PPO'],
            xlabel_y_coordinate=-0.16,
            ax=ax,
            subfigure_width=7,
            row_height=2,
            xlabel='Human Normalized Score')

        if i % 3 == 1:
            axes[i-1].set_yticklabels(['PROPS', 'PPO-Privileged', 'PPO-Buffer', 'PPO'], fontsize='large')
        else:
            axes[i-1].set_yticklabels(['', '', '', '', ], fontsize='large')

    fig.supxlabel('IQM Return', fontsize='large')

    plt.tight_layout()
    # Push plots down to make room for the the legend
    fig.subplots_adjust(left=0.2, top=0.9)
    #
    # # Fetch and plot the legend from one of the subplots.
    # ax = fig.axes[0]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=17)
    #
    save_dir = f'figures'
    save_name = f'return_iqm.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=600)
    #

    plt.show()
