import os

from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file, TIMESTEPS, EVAL_FREQ


def ppo_ros(
        env_id,
        buffer_size,
        lr,
        num_steps,
        update_epochs,
        target_kl,
        ros_lr,
        ros_update_epochs,
        ros_num_steps,
        ros_target_kl,
        ros_clip_coef,
        ros_lambda,
        ros_anneal_lr,
        ros_num_minibatches,
        wandb=0,
        stats=0,
):

    # subdir = f"history_{buffer_size}/num_steps_{num_steps}/num_steps_{ros_num_steps}/lr_{lr}/lr_{ros_lr}/kl_{target_kl}/kl_{ros_target_kl}/epochs_{update_epochs}/epochs_{ros_update_epochs}/clip_{ros_clip_coef}/"
    subdir = f"b_{buffer_size}"

    se_freq = 50
    if 'Swim' in env_id:
        se_freq = 5
    if 'Hop' in env_id:
        se_freq = 20

    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} " \
           f" --total-timesteps {TIMESTEPS[env_id]} --eval-freq {EVAL_FREQ[env_id]}" \
           f" -lr {lr} --num-steps {num_steps} --update-epochs {update_epochs} --anneal-lr 1" \
           f" -b {buffer_size} --ros-num-steps {ros_num_steps} -ros-lr {ros_lr} --ros-update-epochs {ros_update_epochs} --ros-num-minibatches {ros_num_minibatches}" \
           f" --ros-clip-coef {ros_clip_coef} --ros-anneal-lr {ros_anneal_lr}" \
           f" --compute-sampling-error {stats} --se-freq {se_freq} --log-stats 1 " #\
           # f" --cutoff-timesteps {CUTOFFS[env_id]}"

    if ros_lambda:
        args += f" --ros-lambda {ros_lambda} --ros-num-actions 0 --ros-uniform-sampling 0"

    if target_kl:
        args += f" --target-kl {target_kl}"

    if ros_target_kl:
        args += f" --ros-target-kl {ros_target_kl}"


    if wandb:
        args += f" --track 1 --wandb-project-name sweep2"

    if 'Humanoid' in args:
        mem = 0.7
    else:
        mem = 0.4
    disk = 7

    return mem, disk, args

def args_hopper():

    os.makedirs('commands', exist_ok=True)

    TUNED_PARAMS = {
        'Hopper-v4': {
            'buffer_size': 2,
            'num_steps': 2048,
            'ros_num_steps': 512,

            'lr': 1e-4,
            'ros_lr': 1e-3,

            'update_epochs': 10,
            'ros_update_epochs': 16,

            'target_kl': 0.03,
            'ros_target_kl': 0.1,

            'ros_lambda': 0.01,
            'ros_num_minibatches': 16,
            'ros_clip_coef': 0.3,
            'ros_anneal_lr': 0
        },
        'Walker2d-v4': {
            'buffer_size': 2,
            'num_steps': 2048,
            'ros_num_steps': 256,

            'lr': 1e-3,
            'ros_lr': 1e-3,

            'update_epochs': 10,
            'ros_update_epochs': 16,

            'target_kl': 0.03,
            'ros_target_kl': 0.1,

            'ros_lambda': 0.1,
            'ros_num_minibatches': 16,
            'ros_clip_coef': 0.3,
            'ros_anneal_lr': 0,
        },
        'Ant-v4': {
            'buffer_size': 2,
            'num_steps': 1024,
            'ros_num_steps': 256,

            'lr': 1e-4,
            'ros_lr': 1e-3,

            'update_epochs': 10,
            'ros_update_epochs': 16,

            'target_kl': 0.03,
            'ros_target_kl': 0.03,

            'ros_lambda': 0.1,
            'ros_num_minibatches': 32,
            'ros_clip_coef': 0.3,
            'ros_anneal_lr': 0,
        },
        'Humanoid-v4': {
            'buffer_size': 2,
            'num_steps': 8192,
            'ros_num_steps': 256,

            'lr': 1e-4,
            'ros_lr': 1e-4,

            'update_epochs': 10,
            'ros_update_epochs': 16,

            'target_kl': 0.03,
            'ros_target_kl': 0.03,

            'ros_lambda': 0.1,
            'ros_num_minibatches': 32,
            'ros_clip_coef': 0.3,
            'ros_anneal_lr': 0,
        }
    }

    f = open(f"commands/tuned.txt", "w")

    for env_id in TUNED_PARAMS.keys():
        args = gen_args(
            device='cpu',
            length="short",
            arg_generator=ppo_ros,
            env_id=env_id,
            stats=0,
            **TUNED_PARAMS[env_id]
        )
        write_to_file(f, args)


if __name__ == "__main__":

    args_hopper()


