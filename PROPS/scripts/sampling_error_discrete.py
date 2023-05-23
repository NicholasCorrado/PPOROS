
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

def ppo_ros(env_id, ros_update_epochs, lr_ros, num_steps, buffer_history, stats, total_timesteps, reset_freq,expert):
    if expert:
        subdir = f"expert/history_{buffer_history}/ros_update_epochs_{ros_update_epochs}/lr_ros_{lr_ros}/"
    else:
        subdir = f"random/history_{buffer_history}/ros_update_epochs_{ros_update_epochs}/lr_ros_{lr_ros}/"

    args = f" ppo_ros_discrete.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps}" \
           f" -lr 0 -lr-ros {lr_ros} --ros-update-epochs {ros_update_epochs} -b {buffer_history} --num-steps {num_steps} --compute-sampling-error {stats} --eval-freq 1" \
           f" --update-epochs 0 --ros-reset-freq {reset_freq}"

    if expert:
        args += f' --policy-path policies/{env_id}.zip'
    mem = 1.1
    disk = 7

    return mem, disk, args

def ppo(env_id, num_steps, df, total_timesteps, stats, expert):
    if expert:
        subdir = f"expert/df_{df}"
    else:
        subdir = f"random/df_{df}"

    args = f" ppo_discrete.py --env-id {env_id} -s {subdir} --total-timesteps {df*total_timesteps}" \
           f" -lr 0 --num-steps {df*num_steps} --compute-sampling-error {stats} --eval-freq 1" \
           f" --update-epochs 0"

    if expert:
        args += f' --policy-path policies/{env_id}.zip'

    mem = 1.1
    disk = 7

    return mem, disk, args

def ppo_buffer(env_id, num_steps, buffer_history, stats, total_timesteps, expert):
    if expert:
        subdir = f"expert/history_{buffer_history}"
    else:
        subdir = f"random/history_{buffer_history}"

    args = f" ppo_ros_discrete.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps}" \
           f" -lr 0 --ros 0 -b {buffer_history} --num-steps {num_steps} --compute-sampling-error {stats} --eval-freq 1" \
           f" --update-epochs 0"
        # f" --update-epochs 0 --policy-path policies/{env_id}.zip"
    if expert:
        args += f' --policy-path policies/{env_id}.zip'
    mem = 1.1
    disk = 7

    return mem, disk, args

def write_args():


    env_ids = ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
    f = open(f"args/discrete.txt", "w")

    num_collects = 32

    for env_id in env_ids:
        for buffer_history in []:
            for num_steps in [256]:
                for expert in [0,1]:
                    args = gen_args(
                        device='cpu',
                        length="short",
                        arg_generator=ppo_buffer,
                        env_id=env_id,
                        num_steps=num_steps,
                        buffer_history=buffer_history,
                        stats=1,
                        total_timesteps=num_steps * num_collects,
                        expert=expert
                    )
                    write_to_file(f, args)

    for env_id in env_ids:
        for num_steps in [256]:
            for df in [1,2,4,8,16,32]:
                for expert in [0,1]:

                    args = gen_args(
                        device='cpu',
                        length="short",
                        arg_generator=ppo,
                        env_id=env_id,
                        num_steps=num_steps,
                        df=df,
                        total_timesteps=int(num_steps*num_collects),
                        stats=1,
                        expert=expert
                    )
                    write_to_file(f, args)

    for env_id in env_ids:
        for lr_ros in [1e-2, 1e-3, 1e-4, 1e-5]:
            for ros_update_epochs in [64,128,256,512,1024]:
                for buffer_history in [16]:
                    for reset_freq in [1]:
                        for num_steps in [256]:
                            for expert in [0, 1]:

                                args = gen_args(
                                    device='cpu',
                                    length="short",
                                    arg_generator=ppo_ros,
                                    env_id=env_id,
                                    lr_ros=lr_ros,
                                    ros_update_epochs=ros_update_epochs,
                                    num_steps=num_steps,
                                    buffer_history=buffer_history,
                                    stats=1,
                                    total_timesteps=num_steps*num_collects,
                                    reset_freq=reset_freq,
                                    expert=expert
                                )
                                write_to_file(f, args)




if __name__ == "__main__":

    write_args()

    save_dir = 'results'
    qf_ler = 1e-6
    save_subdir = f'qf_lr_{qf_lr}'


