# Proximal Robust On-Policy Sampling (PROPS)

## installation
```commandline
conda create -n props python=3.9
conda activate props
pip install -e PROPS

conda install pytorch cpuonly -c pytorch
conda install stable-baselines3
conda install tensorboard
conda install mujoco
conda install imageio
pip install gymnasium
pip install pyyaml
```

## Code Description

We build our implementation of PROPS on top of CleanRL's implementation of PPO.
The core RL training code is in `ppo_props_continuous.py` file, and utility functions are in `utils.py`.
The core PROPS implementation is in the `update_props` function inside `ppo_props_continuous.py`. 
The PROPS update implementation is very similar to the PPO update implementation.

The `policies` directory contains pretrained expert policies for all six MuJoCo tasks.

## Hyperparameters

We provide tuned hyperparameters for all experiments in the `hyperparameters` directory.

## Running Experiments

All commands required to reproduce experiments in the paper's main body and appendix can be found in the `commands` directory.

* `rl_props.txt`: RL training with PROPS and on-policy sampling.
* `rl_props_ablation_b.txt`: RL training with PROPS with larger buffer sizes.
* `rl_props_ablation_clip_regularization.txt`: RL training with PROPS without clipping/regularization.
* `se_fixed_target_on_policy.txt`: On-policy sampling error with a fixed target policy.
* `se_fixed_target_props.txt`: PROPS sampling error with a fixed target policy.
* `se_fixed_target_props.txt`: PROPS sampling error with a fixed target policy without clipping/regularization.
* `se_fixed_target_ros.txt`: ROS sampling error with a fixed target policy.

By default, results are saved to `results/<env_id>/<algo>/` in numpy archive format (`.npz`).

## Plotting data

Data and plotting scripts for the NeurIPS 2023 submission are in `plotting/neurips/`.
```commandline
conda create -n props python=3.9
conda activate props
pip install -e PROPS

cd PROPS/plotting/neurips
python return.py
```