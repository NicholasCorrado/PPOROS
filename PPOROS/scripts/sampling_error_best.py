def write_to_file(f, args):
    args = args.replace(' ', '*')
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
    args = default_args + other_args

    return args

def ppo_ros(env_id, ros_update_epochs, ros_lr, num_steps, buffer_history, stats, total_timesteps, reset_freq, expert, ros_num_steps):
    if expert:
        subdir = f"expert/history_{buffer_history}/num_steps_{num_steps}/num_steps_{ros_num_steps}/lr_{ros_lr}/epochs_{ros_update_epochs}"
    else:
        subdir = f"random/history_{buffer_history}/num_steps_{num_steps}/num_steps_{ros_num_steps}/lr_{ros_lr}/epochs_{ros_update_epochs}"

    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-episodes 0" \
           f" -lr 0 -ros-lr {ros_lr} --ros-update-epochs {ros_update_epochs} -b {buffer_history} --num-steps {num_steps} --compute-sampling-error {stats} --eval-freq 1" \
           f" --update-epochs 0 --ros-reset-freq {reset_freq} --ros-num-steps {ros_num_steps}"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 5
    disk = 8

    return mem, disk, args

def ppo(env_id, num_steps, df, total_timesteps, stats, expert):
    if expert:
        subdir = f"expert/df_{df}"
    else:
        subdir = f"random/df_{df}"

    args = f" ppo_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {df*total_timesteps} --eval-episodes 0" \
           f" -lr 0 --num-steps {df*num_steps} --compute-sampling-error {stats} --eval-freq 1" \
           f" --update-epochs 0"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'

    mem = 1
    disk = 8

    return mem, disk, args

def ppo_buffer(env_id, num_steps, buffer_history, stats, total_timesteps, expert):
    if expert:
        subdir = f"expert/history_{buffer_history}"
    else:
        subdir = f"random/history_{buffer_history}"

    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-episodes 0" \
           f" -lr 0 --ros 0 -b {buffer_history} --num-steps {num_steps} --compute-sampling-error {stats} --eval-freq 1" \
           f" --update-epochs 0"
    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 5
    disk = 8

    return mem, disk, args


BEST = {
    'Ant-v4': {
        # 'ros_lr': 1e-4,
        # 'ros_epochs': 20
        'ros_lr': 1e-4,
        'ros_epochs': 30
    },
    'Humanoid-v4': {
        'ros_lr': 1e-5,
        'ros_epochs': 10
    },
    'Walker2d-v4': {
        'ros_lr': 1e-3,
        'ros_epochs': 20
    },
    'Hopper-v4': {
        'ros_lr': 1e-3,
        'ros_epochs': 10
    },
    'HalfCheetah-v4': {
        'ros_lr': 1e-3,
        'ros_epochs': 20
    },
    'Swimmer-v4': {
        'ros_lr': 1e-5,
        'ros_epochs': 20
    },
}

def write_args():


    env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    # env_ids = ['Hopper-v4', 'Walker2d-v4']
    env_ids = ['Ant-v4', 'Humanoid-v4', 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', ]

    f = open(f"args/discrete.txt", "w")

    num_collects = 32

    # for env_id in env_ids:
    #     for num_steps in [1024]:
    #         for df in [1,2,4,8,16]:
    #             for expert in [0]:
    #                 args = gen_args(
    #                     device='cpu',
    #                     length="short",
    #                     arg_generator=ppo,
    #                     env_id=env_id,
    #                     num_steps=num_steps,
    #                     df=df,
    #                     total_timesteps=int(num_steps*num_collects),
    #                     stats=1,
    #                     expert=expert
    #                 )
    #                 write_to_file(f, args)
    #

    for expert in [1]:
        for env_id in env_ids:
            for buffer_history in [16]:
                for num_steps in [1024]:
                    args = gen_args(
                        device='cuda',
                        length="short",
                        arg_generator=ppo_buffer,
                        env_id=env_id,
                        num_steps=num_steps,
                        buffer_history=buffer_history,
                        stats=1,
                        total_timesteps=num_steps * num_collects,
                        expert=expert,
                    )
                    # write_to_file(f, args)

                    ros_lr = BEST[env_id]['ros_lr']
                    ros_update_epochs = BEST[env_id]['ros_epochs']

                    args = gen_args(
                        device='cuda',
                        length="short",
                        arg_generator=ppo_ros,
                        env_id=env_id,
                        ros_lr=ros_lr,
                        ros_update_epochs=ros_update_epochs,
                        num_steps=num_steps,
                        buffer_history=buffer_history,
                        stats=1,
                        total_timesteps=num_steps*num_collects,
                        reset_freq=1,
                        expert=expert,
                        ros_num_steps=256
                    )
                    write_to_file(f, args)

if __name__ == "__main__":

    write_args()

