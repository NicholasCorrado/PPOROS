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

def ppo_ros(env_id, total_timesteps, stats, num_steps, buffer_history, expert, ros_num_steps, ros_update_epochs, ros_lr, ros_target_kl,
            ros_lambda):
    if expert:
        subdir = f"expert/l_{ros_lambda}/b_{buffer_history}/s_{num_steps}/s_{ros_num_steps}/lr_{ros_lr}/kl_{ros_target_kl}/e_{ros_update_epochs}"
    else:
        subdir = f"random/l_{ros_lambda}/b_{buffer_history}/s_{num_steps}/s_{ros_num_steps}/lr_{ros_lr}/kl_{ros_target_kl}/e_{ros_update_epochs}"

    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --ros-num-steps {ros_num_steps} -ros-lr {ros_lr} --ros-update-epochs {ros_update_epochs} --ros-target-kl {ros_target_kl} --ros-anneal-lr 0 " \
           f" --compute-sampling-error {stats} --eval-freq 1"

    if ros_lambda:
        args += f" --ros-lambda {ros_lambda}"

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
        subdir = f"expert/b_{buffer_history}/s_{num_steps}"
    else:
        subdir = f"random/b_{buffer_history}/s_{num_steps}"

    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-episodes 0" \
           f" -lr 0 --ros 0 -b {buffer_history} --num-steps {num_steps} --compute-sampling-error {stats} --eval-freq 1" \
           f" --update-epochs 0 --ros-anneal-lr 0"
    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args


ACTION_DIM = {
    'Ant-v4': 8,
    'Humanoid-v4': 19,
    'Walker2d-v4': 6,
    'Hopper-v4': 3,
    'HalfCheetah-v4': 6,
    'Swimmer-v4': 2,
}

def write_args():


    env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    # env_ids = ['Hopper-v4', 'Walker2d-v4']
    env_ids = [ 'Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Swimmer-v4', 'Ant-v4', 'Humanoid-v4']

    # env_ids = ['Humanoid-v4', 'Swimmer-v4']
    # env_ids = ['Ant-v4', 'Humanoid-v4',  'Swimmer-v4', ]

    # env_ids = ['Hopper-v4', 'HalfCheetah-v4','Walker2d-v4',  ]
    # env_ids = ['Hopper-v4']

    f = open(f"commands/se.txt", "w")

    num_collects = 16

    for expert in [1]:
        for buffer_history in [16]:
            for num_steps in [1024, 2048, 4096]:
                num_collects = buffer_history*2
                for env_id in env_ids:
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
                    write_to_file(f, args)

if __name__ == "__main__":

    write_args()


