import copy
import json
import os
import warnings
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from rliable.plot_utils import _annotate_and_decorate_axis
from torch import optim

YLIMS = {
    'Hopper-v4': (0, 4000),
    'HalfCheetah-v4': (0, 5000),
    'Ant-v4': (0, 6000),
    'Walker2d-v4': (0, 5000),
    'Humanoid-v4': (0, 6000),
    'Swimmer-v4': (-50, 200),
    'InvertedPendulum-v4': (0, 1200),
    'InvertedDoublePendulum-v4': (0, 11000),
    'Pendulum-v1': (-1200, -100),
    'Reacher-v4': (-70, 0),
}

def get_data(results_dir, field_name='returns', filename='evaluations.npz'):

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
    for path in paths[:-1]:

        with np.load(path) as data:

            vals = data[field_name]
            if len(vals.shape) > 1:
                avg_vals = np.average(vals, axis=1)
            else:
                avg_vals = vals

            results.append(avg_vals)
            try:
                timesteps = data['timesteps']
            except:
                timesteps = data['t']

    return timesteps, np.array(results)
    # n = 490
    # return timesteps[:n], np.array(results)[:n]

def plot_sample_efficiency_curve(frames,
                                 point_estimates,
                                 interval_estimates,
                                 algorithms,
                                 colors=None,
                                 linestyles=None,
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
    color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))
  if linestyles is None:
      linestyles = {}

  for algorithm in algorithms:
    metric_values = point_estimates[algorithm]
    lower, upper = interval_estimates[algorithm]
    ax.plot(
        frames[algorithm],
        metric_values,
        color=colors[algorithm],
        marker='',
        linewidth=kwargs.pop('linewidth', 2),
        linestyle=linestyles.pop(algorithm, '-'),
        label=algorithm)
    ax.fill_between(
        frames[algorithm], y1=lower, y2=upper, color=colors[algorithm], alpha=0.2)

  return _annotate_and_decorate_axis(
      ax,
      xlabel=xlabel,
      ylabel=ylabel,
      labelsize=labelsize,
      ticklabelsize=ticklabelsize,
      **kwargs)

def get_reinforce_on_policy_data(env_id):
    results_dict = {}
    timesteps_dict = {}
    colors_dict = {}

    for algo in ['reinforce_on_policy']:
        for b in [1]:
            results_dir = f"../../chtc/results/reinforce_tuned/results/{env_id}/{algo}" \
                          f"/b_{b}/"
            timesteps, results = get_data(results_dir=results_dir)

            # A warning will be raised when we fail to load from `results_dir`. Skip these failures.
            if len(results) > 0:
                # mask = timesteps < TIMESTEPS[env_id]
                # timesteps = timesteps[mask]
                # results = results[:, mask]

                key = f"{algo}"
                results_dict[key] = results
                timesteps_dict[key] = timesteps
                colors_dict[key] = 'k'

    return results_dict, timesteps_dict, colors_dict