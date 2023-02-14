
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

def arg_generator(env_id, lr, lr_ros, num_steps, buffer_history):
    subdir = f"lr_ros_{lr_ros}/history_{buffer_history}"
    args = f" --env-id {env_id} -s {subdir} --total-timesteps 500000" \
           f" -lr {lr} -lr-ros {lr_ros} -b {buffer_history} --num-steps {num_steps}"

    mem = 1
    disk = 6

    return mem, disk, args

def arg_generator_ppo(env_id, lr, num_steps, total_timesteps):
    subdir = f"df_{num_steps//256}"
    args = f" --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps}" \
           f" -lr {lr} --num-steps {num_steps}"

    mem = 1
    disk = 6

    return mem, disk, args

def write_args():


    env_ids = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']

    f = open(f"args/ppo_ros.txt", "w")
    for env_id in env_ids:
        for lr in [1e-4]:
            for lr_ros in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
                for buffer_history in [2,4,8]:
                    for num_steps in [256]:
                        args = gen_args(
                            device='cpu',
                            length="short",
                            arg_generator=arg_generator,
                            env_id=env_id,
                            lr=lr,
                            lr_ros=lr_ros,
                            num_steps=num_steps,
                            buffer_history=buffer_history
                        )
                        write_to_file(f, args)

    f = open(f"args/ppo.txt", "w")
    for env_id in env_ids:
        for lr in [1e-4]:
            for df in [1,2,4,8]:
                args = gen_args(
                    device='cpu',
                    length="short",
                    arg_generator=arg_generator_ppo,
                    env_id=env_id,
                    lr=lr,
                    num_steps=int(256*df),
                    total_timesteps=int(500e3*df)
                )
                write_to_file(f, args)


if __name__ == "__main__":

    write_args()


