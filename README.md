# Proximal Robust On-Policy Sampling (PROPS)

## installation
```commandline
conda create -n props python=3.9
conda activate props
pip install -e PPOROS

CHANGE PPOROS TO PROPS EVERYWHERE

conda install pytorch cpuonly -c pytorch
conda install stable-baselines3
conda install tensorboard
conda install mujoco
conda install imageio
pip install gymnasium
pip install pyyaml
```

## Code Description

The core PROPS implementation is in the `update_props` function inside `ppo_props_continuous.py`. 
This function is 

## Reproducing Experiments

All commands required to reproduce experiments in the paper's main body and appendix can be found in the `commands` directory.

* `rl_props.txt`: RL training with PROPS and on-policy sampling
* `rl_props_ablation_b.txt`: RL training with PROPS with larger buffer sizes.
* `rl_props_ablation_clip_regularization.txt`: RL training with PROPS without clipping/regularization.
* `se_fixed_target_on_policy.txt`: On-policy sampling error with a fixed target policy.
* `se_fixed_target_props.txt`: PROPS sampling error with a fixed target policy.
* `se_fixed_target_props.txt`: PROPS sampling error with a fixed target policy without clipping/regularization.
* `se_fixed_target_ros.txt`: ROS sampling error with a fixed target policy.

By default, results are saved to `results/<env_id>/<algo>/`.


