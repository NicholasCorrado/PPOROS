import os

import numpy as np
import pandas
import seaborn
from matplotlib import pyplot as plt

from PROPS.plotting.utils import get_paths, plot, load_data

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from rliable.plot_utils import _decorate_axis, _annotate_and_decorate_axis




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

            avgs.append(avg[[-1]])

    return t, np.array(avgs)

def get_data(path_dict):
  results = {}
  timesteps = {}
  for agent, info in path_dict.items():
    paths = info['paths']

    t, avgs = load_data(paths, name='returns')
    results[agent] = avgs

  return results

if __name__ == "__main__":

    seaborn.set_theme(style='whitegrid', palette='colorblind')

    env_ids = ['TwoStep-v0']


    root_dir = 'condor/results'
    # root_dir = 'lr_sweep/results'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1 * 4, 1 * 5))
    i = 1

    for env_id in env_ids:
        # ax = plt.subplot(1, 2, i)
        i+=1

        path_dict_all = {}
        ### PROPS ##################################################################################################
        key = rf'Adaptive Sampling'
        algo = 'ppo_props'
        results_dir = f'{root_dir}/{env_id}/{algo}/b_1'
        path_dict_aug = get_paths(
            results_dir=results_dir,
            key=key,
            evaluations_name='evaluations')
        if len(path_dict_aug[key]['paths']) > 0:
            path_dict_all.update(path_dict_aug)

        ### PROPS ##################################################################################################
        # key = rf'PROPS, $b=2$'
        # algo = 'ppo_props'
        # results_dir = f'{root_dir}/{env_id}/{algo}/b_2'
        # path_dict_aug = get_paths(
        #     results_dir=results_dir,
        #     key=key,
        #     evaluations_name='evaluations')
        # if len(path_dict_aug[key]['paths']) > 0:
        #     path_dict_all.update(path_dict_aug)

        ### PPO #########################################################################################
        key = rf'On-Policy Sampling'
        algo = 'ppo_buffer'
        results_dir = f'{root_dir}/{env_id}/{algo}/b_1'
        path_dict_aug = get_paths(
            results_dir=results_dir,
            key=key,
            evaluations_name='evaluations')
        if len(path_dict_aug[key]['paths']) > 0:
            path_dict_all.update(path_dict_aug)

        # key = rf'PPO, $b=2$'
        # algo = 'ppo_buffer'
        # results_dir = f'{root_dir}/{env_id}/{algo}/b_2'
        # path_dict_aug = get_paths(
        #     results_dir=results_dir,
        #     key=key,
        #     evaluations_name='evaluations')
        # if len(path_dict_aug[key]['paths']) > 0:
        #     path_dict_all.update(path_dict_aug)

        plot(path_dict_all, name='returns')
        plt.title(f'PPO Training Curves', fontsize=20)
        plt.xlabel('Timestep', fontsize=20)
        plt.ylabel('Average Return', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(alpha=0.2)
        # plt.ylim(0.7,1.05)
        # plt.xscale('log')
        # Use scientific notation for x-axis
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # set fontsize of 1e6
        ax.xaxis.get_offset_text().set_fontsize(20)
        plt.tight_layout()

        # fig.subplots_adjust(top=0.8)
        # fig.legend(loc='upper center', ncols=1, fontsize=20)

        # fig.subplots_adjust(top=0.8)
        ax.legend(loc='lower right', ncols=1, fontsize=14)



        save_dir = f'figures'
        save_name = f'discrete_return.png'
        os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(f'{save_dir}/{save_name}')
        plt.savefig(f'toy_return', dpi=300)
        plt.show()


        data_dict = get_data(path_dict_all)
        outcome_dict = {}
        for k, v in data_dict.items():
            N = len(v)
            # v = v.astype(int)
            outcome_dict[k] = {
                'max': np.sum(v == 2)/N,
                'min': np.sum(v == 0.5)/N,
                'subopt': np.sum(v == 1)/N,
            }
            outcome_dict[k] = [np.sum(v == 2)/N, np.sum(v == 0.5)/N, np.sum(v == 1)/N]

            print(outcome_dict)
        thresholds = np.linspace(0.5, 2.1, 101)

        algorithms = list(data_dict.keys())
        penguins = seaborn.load_dataset("penguins")
        pandas.DataFrame()

        trajectories = [r'$\tau_{MAX}$', r'$\tau_{MIN}$', r'$\tau_{SUBOPT}$']

        seaborn.set_theme(style='whitegrid', palette='colorblind')

        penguin_means = {
            r'$\tau_{MAX}$': [outcome_dict[algorithms[0]][0], outcome_dict[algorithms[1]][0]],
            r'$\tau_{MIN}$': [outcome_dict[algorithms[0]][1], outcome_dict[algorithms[1]][1]],
            r'$\tau_{SUBOPT}$': [outcome_dict[algorithms[0]][2], outcome_dict[algorithms[1]][2]],
        }

        x = np.arange(2)  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1 * 4, 1 * 5))
        plt.subplot(1,1,1)

        for attribute, measurement in penguin_means.items():
            # if multiplier == 0: continue
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3, fontsize=24)
            multiplier += 1

        algorithms = ['Adaptive\nSampling', 'On-Policy\nSampling']
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Fraction of Agents\n Converging to Each Trajectory', fontsize=20)
        ax.set_title('Trajectory Distribution', fontsize=20)
        ax.set_xticks([0.25, 1.25], algorithms, fontsize=20)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=16)
        ax.set_ylim(0, 1.2)
        ax.xaxis.grid(False)
        plt.tight_layout()

        fig.subplots_adjust(top=0.8)
        fig.legend(loc='upper center', ncols=3, fontsize=13.5)



        save_dir = f'figures'
        save_name = f'traj_dist.png'
        # os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(f'{save_dir}/{save_name}')
        plt.savefig(f'toy_traj_dist', dpi=300)

        plt.show()

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1 * 6, 1 * 5))
        #
        # ax = plt.bar(x=['', 'PPO with \nOn-Policy Sampling'], height=[1, 0.71])
        #
        # colors = seaborn.color_palette('colorblind')
        # for patch, color in zip(ax.patches, colors[:2]):
        #     patch.set_facecolor(color)
        #
        # plt.ylabel('Fraction of Agents Converging\nto the Optimal Policy', fontsize=16)
        # # plt.title('Trajectory Distribution Over 100 Training Runs', fontsize=16)
        # # plt.grid(False)
        # plt.tight_layout()
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # fig.subplots_adjust(top=0.8)
        # fig.legend(loc='upper center', ncols=2, fontsize=18)
        # plt.savefig(f'toy_traj_dist_2', dpi=300)

        # plt.show()

        #
        # plt.bar(x=trajectories, height=)
        # df = pandas.DataFrame({
        #     'Algorithm': algorithms,
        #     'Trajectory': [r'$\tau_{MAX}$', r'$\tau_{MIN}$', r'$\tau_{SUBOPT}$'],
        #     r'$\tau_{MAX}$': [outcome_dict[algorithms[0]]['max'], outcome_dict[algorithms[1]]['max']],
        #     r'$\tau_{MIN}$': [outcome_dict[algorithms[0]]['min'], outcome_dict[algorithms[1]]['min']],
        #     r'$\tau_{SUBOPT}$': [outcome_dict[algorithms[0]]['subopt'], outcome_dict[algorithms[1]]['subopt']],
        # })
        #
        # g = seaborn.catplot(
        #     data=df, kind="bar",
        #     x="Trajectory", y="Probability", hue="Algorithm",
        #     errorbar="sd", palette="colorblind", alpha=.6, height=6
        # )
        # g.despine(left=True)
        # g.set_axis_labels("", "Body mass (g)")
        # g.legend.set_title("")
        # g.show()
        #






        # score_distributions, score_distributions_cis = rly.create_performance_profile(
        #     data_dict, thresholds, reps=500)
        # # Plot score distributions
        # ax = plt.subplot(1, 2, 2)
        # plot_utils.plot_performance_profiles(
        #     score_distributions, thresholds,
        #     performance_profile_cis=score_distributions_cis,
        #     colors=dict(zip(algorithms, seaborn.color_palette('colorblind'))),
        #     xlabel=r'Return $(\tau)$',
        #     # linestyles=linestyles,
        #     # alpha=0,
        #     ax=ax)
        # plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5)
        # plt.ylabel('Fraction of runs\n' r'with score > $\tau$', fontsize='xx-large')
        # plt.xlabel(r'Return $(\tau)$', fontsize='xx-large')
        #
        # plt.tight_layout()
        #
        # # Push plots down to make room for the the legend
        # fig.subplots_adjust(top=0.8)
        #
        # # Fetch and plot the legend from one of the subplots.
        # ax = fig.axes[0]
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=20)
        #
        # plt.savefig(f'2step_results', dpi=300)
        #
        # plt.show()
