import os

from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


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
        ros_mb,
        wandb=0,
        stats=0,
):

    # subdir = f"history_{buffer_size}/num_steps_{num_steps}/num_steps_{ros_num_steps}/lr_{lr}/lr_{ros_lr}/kl_{target_kl}/kl_{ros_target_kl}/epochs_{update_epochs}/epochs_{ros_update_epochs}/clip_{ros_clip_coef}/"
    subdir = f"b_{buffer_size}/s_{num_steps}/s_{ros_num_steps}/lr_{lr}/lr_{ros_lr}/kl_{target_kl}/kl_{ros_target_kl}/l_{ros_lambda}/e_{ros_update_epochs}/mb_{ros_mb}/c_{ros_clip_coef}/a_{ros_anneal_lr}"

    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} " \
           f" --total-timesteps {int(200e3)} --eval-freq {10}" \
           f" -lr {lr} --num-steps {num_steps} --update-epochs {update_epochs} --anneal-lr 1" \
           f" -b {buffer_size} --ros-num-steps {ros_num_steps} -ros-lr {ros_lr} --ros-update-epochs {ros_update_epochs} --ros-num-minibatches {ros_mb}" \
           f" --ros-clip-coef {ros_clip_coef} --ros-anneal-lr {ros_anneal_lr}" \
           f" --compute-sampling-error {stats} --log-stats 1 " \
           f" --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}"
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

    env_ids = ['Hopper-v4']
    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for buffer_size in [2]:
            f = open(f"commands/ros_b{buffer_size}.txt", "w")
            for num_steps in [2048]:
                for lr in [1e-4]:
                    for ros_lr in [1e-3, 1e-4]:
                        for ros_update_epochs in [16]:
                            for ros_mb in [8, 16]:
                                for ros_lambda in [0, 0.01, 0.1]:
                                    for ros_num_steps in [512,1024,2048]:
                                        for ros_target_kl in [0.03, 0.05, 0.1]:

                                            if ros_num_steps > num_steps: continue
                                            # if ros_target_kl == 0.05 and ros_lambda < 0.5: continue

                                            update_epochs = 10
                                            target_kl = 0.03

                                            ros_clip_coef = 0.3
                                            ros_anneal_lr = 0

                                            for ros_anneal_lr in [0]:
                                                args = gen_args(
                                                    device='cpu',
                                                    length="short",
                                                    arg_generator=ppo_ros,
                                                    env_id=env_id,
                                                    num_steps=num_steps,
                                                    lr=lr,
                                                    update_epochs=update_epochs,
                                                    target_kl=target_kl,
                                                    ros_lr=ros_lr,
                                                    buffer_size=buffer_size,
                                                    ros_num_steps=ros_num_steps,
                                                    ros_update_epochs=ros_update_epochs,
                                                    ros_mb=ros_mb,
                                                    ros_target_kl=ros_target_kl,
                                                    ros_clip_coef=ros_clip_coef,
                                                    ros_anneal_lr=ros_anneal_lr,
                                                    stats=0,
                                                    ros_lambda=ros_lambda
                                                )
                                                write_to_file(f, args)


if __name__ == "__main__":

    args_hopper()


