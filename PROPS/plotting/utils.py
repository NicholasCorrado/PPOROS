import json
import os
import warnings
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from rliable.plot_utils import _annotate_and_decorate_axis


def get_data(results_dir, field_name='returns', filename='evaluations.npz'):

    try:
        paths = []
        for subdir in os.listdir(results_dir):
            if '_sources' not in subdir:
                paths.append(f'{results_dir}/{subdir}/{filename}')
    except:
        warnings.warn(f'Data not found at path {results_dir}')
        paths = []

    timesteps = None
    results = []
    for path in paths:

        with np.load(path) as data:

            vals = data[field_name]
            if len(vals.shape) > 1:
                avg_vals = np.average(vals, axis=1)
            else:
                avg_vals = vals

            results.append(avg_vals)
            timesteps = data['timesteps']


    return timesteps, np.array(results)




def plot_sample_efficiency_curve(frames,
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
    color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))

  for algorithm in algorithms:
    metric_values = point_estimates[algorithm]
    lower, upper = interval_estimates[algorithm]
    ax.plot(
        frames[algorithm],
        metric_values,
        color=colors[algorithm],
        marker='',
        linewidth=kwargs.pop('linewidth', 2),
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
