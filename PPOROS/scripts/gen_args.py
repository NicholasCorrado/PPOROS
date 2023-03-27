
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

def ppo_ros(env_id, lr, lr_ros, num_steps, buffer_history):
    subdir = f"history_{buffer_history}/lr_{lr}/lr_ros_{lr_ros}/"
    args = f" ppo_ros_discrete.py --env-id {env_id} -s {subdir} --total-timesteps 500000" \
           f" -lr {lr} -lr-ros {lr_ros} -b {buffer_history} --num-steps {num_steps}"

    mem = 1
    disk = 7

    return mem, disk, args

def ppo(env_id, lr, num_steps, total_timesteps):
    subdir = f"df_{num_steps//256}/lr_{lr}"
    args = f" ppo_discrete.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps}" \
           f" -lr {lr} --num-steps {num_steps}"

    mem = 1
    disk = 7

    return mem, disk, args

def ppo_buffer(env_id, lr, num_steps, buffer_history):
    subdir = f"ppo_buffer/history_{buffer_history}/lr_{lr}"
    args = f" ppo_ros_discrete.py --env-id {env_id} -s {subdir} --total-timesteps 500000" \
           f" -lr {lr} --ros 0 -b {buffer_history} --num-steps {num_steps}"

    mem = 1
    disk = 7

    return mem, disk, args

def write_args():


    env_ids = ['CartPole-v1', 'Acrobot-v1']#, 'LunarLander-v2']

    # f = open(f"args/ppo_ros_discrete.txt", "w")
    # for env_id in env_ids:
    #     for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
    #         for lr_ros in [1e-5, 1e-6]:
    #             for buffer_history in [2,4,8,16]:
    #                 for num_steps in [256]:
    #                     args = gen_args(
    #                         device='cpu',
    #                         length="short",
    #                         arg_generator=ppo_ros,
    #                         env_id=env_id,
    #                         lr=lr,
    #                         lr_ros=lr_ros,
    #                         num_steps=num_steps,
    #                         buffer_history=buffer_history
    #                     )
    #                     write_to_file(f, args)
    #
    f = open(f"args/ppo_buffer_discrete.txt", "w")
    for env_id in env_ids:
        for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
            for buffer_history in [2,4,8,16]:
                for num_steps in [256]:
                    args = gen_args(
                        device='cpu',
                        length="short",
                        arg_generator=ppo_buffer,
                        env_id=env_id,
                        lr=lr,
                        num_steps=num_steps,
                        buffer_history=buffer_history
                    )
                    write_to_file(f, args)

    f = open(f"args/ppo_discrete.txt", "w")
    for env_id in env_ids:
        for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
            for df in [1,2,4,8,16]:
                args = gen_args(
                    device='cpu',
                    length="short",
                    arg_generator=ppo,
                    env_id=env_id,
                    lr=lr,
                    num_steps=int(256*df),
                    total_timesteps=int(500e3*df)
                )
                write_to_file(f, args)


if __name__ == "__main__":

    write_args()


