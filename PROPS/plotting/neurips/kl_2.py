import itertools
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths, plot, load_data, get_line_styles

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from rliable.plot_utils import _decorate_axis, _annotate_and_decorate_axis

def get_data(path_dict):
  returns = {}
  timesteps = {}
  for agent, info in path_dict.items():
    paths = info['paths']

    t, avgs = load_data(paths, name='returns')
    returns[agent] = avgs
    timesteps[agent] = t

  return returns, timesteps


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

def plot_sample_efficiency_curve(frame_dict,
                                 point_estimates,
                                 interval_estimates,
                                 algorithms,
                                 colors=None,
                                 color_palette='colorblind',
                                 figsize=(7, 5),
                                 xlabel=r'Number of Frames (in millions)',
                                 ylabel='Aggregate Human Normalized Score',
                                 ax=None,
                                 labelsize='xx-large',
                                 ticklabelsize='xx-large',
                                 **kwargs):
  """Plots an aggregate metric with CIs as a function of environment frames.

  Args:
    frames: Array or list containing environment frames to mark on the x-axis.
    point_estimates: Dictionary mapping algorithm to a list or array of point
      estimates of the metric corresponding to the values in `frames`.
    interval_estimates: Dictionary mapping algorithms to interval estimates
      corresponding to the `point_estimates`. Typically, consists of stratified
      bootstrap CIs.
    algorithms: List of methods used for plotting. If None, defaults to all the
      keys in `point_estimates`.
    colors: Dictionary that maps each algorithm to a color. If None, then this
      mapping is created based on `color_palette`.
    color_palette: `seaborn.color_palette` object for mapping each method to a
      color.
    figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
      `ax` is None.
    xlabel: Label for the x-axis.
    ylabel: Label for the y-axis.
    ax: `matplotlib.axes` object.
    labelsize: Font size of the x-axis label.
    ticklabelsize: Font size of the ticks.
    **kwargs: Arbitrary keyword arguments.

  Returns:
    `axes.Axes` object containing the plot.
  """
  if ax is None:
    _, ax = plt.subplots(figsize=figsize)
  if algorithms is None:
    algorithms = list(point_estimates.keys())
  if colors is None:
    color_palette = seaborn.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))

  for algorithm in algorithms:
    frames = frame_dict[algorithm]
    metric_values = point_estimates[algorithm]
    lower, upper = interval_estimates[algorithm]
    ls = None
    alpha = 0.2
    # if algorithm in ['PROPS']:
    #     ls = 'None'
    #     alpha = 0

    ax.plot(
        frames,
        metric_values,
        color=colors[algorithm],
        # marker=kwargs.pop('marker', 'o'),
        linewidth=kwargs.pop('linewidth', 2),
        label=algorithm,
        linestyle=ls)
    ax.fill_between(
        frames, y1=lower, y2=upper, color=colors[algorithm], alpha=alpha)

  return _annotate_and_decorate_axis(
      ax,
      xlabel=xlabel,
      ylabel=ylabel,
      labelsize=labelsize,
      ticklabelsize=ticklabelsize,
      **kwargs)


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



def plot(save_dict, name, m=100000, success_threshold=None, return_cutoff=-np.inf, times=None):
    i = 0

    palette = seaborn.color_palette("magma_r", n_colors=3)
    # print(os.getcwd())

    for agent, info in save_dict.items():
        paths = info['paths']
        x_scale = info['x_scale']
        max_t = info['max_t']
        avgs = []
        for path in paths:
            u, t, avg = load_data(path, name=name, success_threshold=success_threshold)
            if avg is not None:
                # print(len(avg))
                # if len(avg) < 50:
                #     continue
                if max_t:
                    cutoff = np.where(t <= max_t/x_scale)[0]
                    avg = avg[cutoff]
                    t = t[cutoff]

                elif m:
                    avg = avg[:m]
                avgs.append(avg)
                t_good = t

        if len(avgs) == 0:
            continue
        elif len(avgs) == 1:
            avg_of_avgs = avg
            q05 = np.zeros_like(avg)
            q95 = np.zeros_like(avg)

        else:

            min_l = np.inf
            for a in avgs:
                l = len(a)
                if l < min_l:
                    min_l = l

            if min_l < np.inf:
                for i in range(len(avgs)):
                    avgs[i] = avgs[i][:min_l]

            # avg_of_avgs = np.median(avgs, axis=0)
            avg_of_avgs = np.mean(avgs, axis=0)

            # if avg_of_avgs.mean() > 0: continue
            # print(np.median(avg_of_avgs))
            # if np.median(avg_of_avgs) > 0: continue

            std = np.std(avgs, axis=0)
            N = len(avgs)
            ci = 1 * std / np.sqrt(N) * 1.96
            q05 = avg_of_avgs - ci
            q95 = avg_of_avgs + ci
            # q05 = np.min(avgs, axis=0)
            # q95 = np.max(avgs, axis=0)

            q25 = np.quantile(avgs, 0.25, axis=0)
            q75 = np.quantile(avgs, 0.75, axis=0)

        style_kwargs = get_line_styles(agent)
        style_kwargs['linewidth'] = 2




        style_kwargs['linewidth'] = 1.5

        # if 'PROPS' in agent:
        #     style_kwargs['linestyle'] = '-'
        #     style_kwargs['linewidth'] = 3
        #
        # elif 'ppo_buffer' in agent or 'PPO-Buffer' in agent or 'b=' in agent or 'Buffer' in agent:
        #     style_kwargs['linestyle'] = '--'
        # elif 'ppo,' in agent or 'PPO,' in agent or 'PPO with' in agent or 'PPO' == agent:
        #     style_kwargs['linestyle'] = ':'
        # elif 'Priv' in agent:
        #     style_kwargs['linestyle'] = '-.'

        if '0.0001' in agent:
            style_kwargs['linestyle'] = '--'

        l = len(avg_of_avgs)
        # print(l)
        print(agent, avg_of_avgs[-1], q05[-1], q95[-1])

        try:
            times = info['times']
            x = times
        except:
            x = t_good * x_scale
            if t is None:
                x = np.arange(len(avg_of_avgs))
            if m:
                x = x[:m]
                avg_of_avgs = avg_of_avgs[:m]
                q05 = q05[:m]
                q95 = q95[:m]

        if '0.03' in agent:
            idx = 0
        if '0.05' in agent:
            idx = 1
        if '0.1' in agent:
            idx = 2
        color = palette[idx]

        plt.plot(x[:l], avg_of_avgs, label=agent, **style_kwargs, color=color)
        if style_kwargs['linestyle'] == 'None':
            plt.fill_between(x[:l], q05, q95, alpha=0, color=color)
        else:
            plt.fill_between(x[:l], q05, q95, alpha=0.1, color=color)
        # plt.fill_between(x, q05, q95, alpha=0.2, color=style_kwargs['color'])

        i += 1
    # return fig

if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid')
    env_ids = ['Swimmer-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4']
    # env_ids = ['Humanoid-v4']
    env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4']

    fig, axes = plt.subplots(1, 3, figsize=(4*4.5,1*4.5))
    axes = axes.flatten()
    i = 0
    for env_id in env_ids:
        i+=1
        ax = axes[i-1]

        path_dict_all = {}
        print(env_id)

        algo = 'ppo_ros'
        ns = [1024, 2048, 4096, 8192]
        lrs = [1e-3, 1e-4]
        ros_lrs = [1e-3, 1e-4, 1e-5]
        for b in [2]:
            for s in [2048]:
                for lr in lrs:
                    for rlr in [1e-3]:
                        for ros_update_epochs in [16]:
                            for ros_mb in [16]:
                                for l in [0.1]:
                                    for rs in [256]:
                                        for rkl in [0.03, 0.05, 0.1]:

                                            root = '/Users/nicholascorrado/code/PPOROS_data_final/PPOROS/PPOROS/plotting/5_08/rsweep'
                                            root = '/Users/nicholascorrado/code/PPOROS_data_final/PPOROS/PPOROS/plotting/5_07/r_1'

                                            results_dir = f'{root}/results/{env_id}/{algo}/b_{b}/s_{s}/s_{rs}/lr_{lr}/lr_{rlr}/kl_0.03/kl_{rkl}/l_{l}/e_16/mb_{ros_mb}/c_0.3/a_0'

                                            key = rf'PROPS: {s},{rs}; {lr},{rlr}; {rkl}; {l}, {ros_mb}'
                                            key = rf'lr={rlr}, $\lambda$={l}'
                                            # key = rf'$\lambda$={l}'
                                            key = f'lr={rlr}, ' + r'$\delta_{PROPS}$' + rf'$={rkl}$'

                                            path_dict_aug = get_paths(
                                                results_dir=results_dir,
                                                key=key,
                                                max_t=int(2e6),
                                                evaluations_name='evaluations')
                                            if len(path_dict_aug[key]['paths']) > 0:
                                                path_dict_all.update(path_dict_aug)

        for b in [2]:
            for s in [2048]:
                for lr in lrs:
                    for rlr in [1e-4]:
                        for ros_update_epochs in [16]:
                            for ros_mb in [16]:
                                for l in [0.1]:
                                    for rs in [256]:
                                        for rkl in [0.03, 0.05, 0.1]:

                                            root = '/Users/nicholascorrado/code/PPOROS_data_final/PPOROS/PPOROS/plotting/5_08/rsweep'
                                            root = '/Users/nicholascorrado/code/PPOROS_data_final/PPOROS/PPOROS/plotting/5_07/r_1'
                                            if env_id == 'Walker2d-v4':
                                                root = '/Users/nicholascorrado/code/PPOROS_data_final/PPOROS/PPOROS/plotting/5_05/ros_b2'
                                                ros_mb = 32

                                            results_dir = f'{root}/results/{env_id}/{algo}/b_{b}/s_{s}/s_{rs}/lr_{lr}/lr_{rlr}/kl_0.03/kl_{rkl}/l_{l}/e_16/mb_{ros_mb}/c_0.3/a_0'

                                            key = rf'PROPS: {s},{rs}; {lr},{rlr}; {rkl}; {l}, {ros_mb}'
                                            key = rf'lr={rlr}, $\lambda$={l}'
                                            # key = rf'$\lambda$={l}'
                                            key = f'lr={rlr}, ' + r'$\delta_{PROPS}$' + rf'$={rkl}$'

                                            path_dict_aug = get_paths(
                                                results_dir=results_dir,
                                                key=key,
                                                max_t=int(2e6),
                                                evaluations_name='evaluations')
                                            if len(path_dict_aug[key]['paths']) > 0:
                                                path_dict_all.update(path_dict_aug)


        return_dict, timestep_dict = get_data(path_dict_all)


        algorithms = list(return_dict.keys())
        ale_frames_scores_dict = {algorithm: score for algorithm, score
                                  in return_dict.items()}
        iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., [frame]])
                                       for frame in range(scores.shape[-1])])
        iqm_scores, iqm_cis = rly.get_interval_estimates(
            ale_frames_scores_dict, iqm, reps=5)
        plot_sample_efficiency_curve(
            timestep_dict, iqm_scores, iqm_cis, algorithms=algorithms,
            ax=axes[i - 1],
            xlabel=r'Timestep',
            ylabel='')
        plt.title(f'{env_id}', fontsize='xx-large')
        if i % 3 == 1:
            plt.ylabel('IQM Return', fontsize='xx-large')
        #
        # # Use scientific notation for x-axis
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # set fontsize of 1e6
        ax.xaxis.get_offset_text().set_fontsize('xx-large')
        #
        plt.tight_layout()
        # plt.show()

    plt.tight_layout()
    # Push plots down to make room for the the legend
    fig.subplots_adjust(left=0.1, top=0.88)
    #
    # # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='xx-large')
    #
    save_dir = f'figures'
    save_name = f'kl_2.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=300)
    #

    plt.show()

