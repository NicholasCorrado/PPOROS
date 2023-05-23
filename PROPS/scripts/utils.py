# TIMESTEPS = {
#     'Hopper-v4': int(2e6),
#     'HalfCheetah-v4': int(5e6),
#     'Ant-v4': int(6e6),
#     'Walker2d-v4': int(5e6),
#     'Humanoid-v4': int(6e6),
#     'Swimmer-v4': int(2e6),
#     'InvertedPendulum-v4': int(300e3),
#     'InvertedDoublePendulum-v4': int(300e3),
# }
TIMESTEPS = {
    'Hopper-v4': int(1e6),
    'HalfCheetah-v4': int(2e6),
    'Ant-v4': int(4e6),
    'Walker2d-v4': int(2e6),
    'Humanoid-v4': int(6e6),
    'Swimmer-v4': int(1e6),
    'InvertedPendulum-v4': int(300e3),
    'InvertedDoublePendulum-v4': int(300e3),
}

EVAL_FREQ = {
    'Hopper-v4': 10,
    'HalfCheetah-v4': 10,
    'Ant-v4': 10,
    'Walker2d-v4': 10,
    'Humanoid-v4': 10,
    'Swimmer-v4': 10,
    'InvertedPendulum-v4': 1,
    'InvertedDoublePendulum-v4': 1,
}

def write_to_file(f, args):
    # args = args.replace(' ', '*')
    print(args)
    f.write(args + "\n")

def gen_args(device, length, arg_generator, **kwargs):
    assert (device == 'cpu') or (device == 'cuda')
    if device == 'cuda':
        device_args = f'\"{length}\",--device cuda'
    else:
        device_args = f'--device cpu'

    mem, disk, other_args = arg_generator(**kwargs)
    default_args = f"{mem},{disk},"
    default_args = ""
    args = default_args + other_args

    return args

CUTOFFS = {
    'Swimmer-v4': int(1e6),
    'Hopper-v4': int(5e5),
    'HalfCheetah-v4': int(1e6),
    'Walker2d-v4': int(1.5e6),
    'Ant-v4': int(2e6),
    'Humanoid-v4': int(4e6),
}

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
        cutoff=0,
):

    # subdir = f"history_{buffer_size}/num_steps_{num_steps}/num_steps_{ros_num_steps}/lr_{lr}/lr_{ros_lr}/kl_{target_kl}/kl_{ros_target_kl}/epochs_{update_epochs}/epochs_{ros_update_epochs}/clip_{ros_clip_coef}/"
    subdir = f"b_{buffer_size}/s_{num_steps}/s_{ros_num_steps}/lr_{lr}/lr_{ros_lr}/kl_{target_kl}/kl_{ros_target_kl}/l_{ros_lambda}/e_{ros_update_epochs}/mb_{ros_mb}/c_{ros_clip_coef}/a_{ros_anneal_lr}"
    subdir = f"b_{buffer_size}"
    se_freq = 50
    if 'Swim' in env_id:
        se_freq = 5
    if 'Hop' in env_id:
        se_freq = 20

    args = f"python ppo_props_continuous.py --env-id {env_id} -s {subdir}" \
           f" --total-timesteps {TIMESTEPS[env_id]} --eval-freq {EVAL_FREQ[env_id]}" \
           f" -lr {lr} --num-steps {num_steps} --update-epochs {update_epochs} --anneal-lr 1" \
           f" -b {buffer_size} --ros 1 --ros-num-steps {ros_num_steps} -ros-lr {ros_lr} --ros-update-epochs {ros_update_epochs} --ros-num-minibatches {ros_mb}" \
           f" --ros-clip-coef {ros_clip_coef} --ros-anneal-lr {ros_anneal_lr}" \
           f" --log-stats 1 --se {stats} --se-lr 1e-3 --se-epochs 100 " #\

    if cutoff:
      args += f" --cutoff-timesteps {CUTOFFS[env_id]} "

    if ros_lambda:
        args += f" --ros-lambda {ros_lambda}"

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
    disk = 8

    return mem, disk, args

def ppo(
        env_id,
        total_timesteps,
        lr,
        num_steps,
        update_epochs,
        target_kl,
        stats,
        save_policy=False,
):
    # subdir = f"df_{num_steps//2048}/lr_{lr}/kl_{target_kl}"
    subdir = f""
    args = f"ppo_continuous.py --env-id {env_id} -total-timesteps {total_timesteps} --eval-freq {EVAL_FREQ[env_id]}" \
           f" -lr {lr} --num-steps {num_steps} --num-minibatches 32 --update-epochs {update_epochs}" \
           f" --compute-sampling-error {stats} --save-policy {save_policy}"

    if target_kl:
        args += f" --target-kl {target_kl}"

    mem = 0.5
    disk = 8

    return mem, disk, args

def ppo_buffer(env_id, lr, num_steps, buffer_size, se, target_kl, epochs, df=1):
    subdir = f""

    se_freq = 20
    if 'Swim' in env_id:
        se_freq = 5
    if 'Hop' in env_id:
        se_freq = 20

    args = f"python ppo_props_continuous.py --env-id {env_id} --total-timesteps {TIMESTEPS[env_id]*df} --eval-freq {EVAL_FREQ[env_id]}" \
           f" -lr {lr} --num-steps {num_steps*2}" \
           f" -b {buffer_size} --ros 0 " \
           f" --se {se} --se-lr 1e-3 --se-epochs 1000 "

    if se:
        args += f' --se-freq {se_freq}'

    if target_kl:
        args += f" --target-kl {target_kl}"

    if 'Humanoid' in env_id:
        mem = 0.7
    else:
        mem = 0.4
    disk = 8

    return mem, disk, args
