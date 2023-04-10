from PPOROS.scripts.utils import ppo_ros

TIMESTEPS = {
    'Hopper-v4': int(2e6),
    'HalfCheetah-v4': int(5e6),
    'Ant-v4': int(6e6),
    'Walker2d-v4': int(5e6),
    'Humanoid-v4': int(6e6),
    'Swimmer-v4': int(2e6),
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

def ppo_ros_mixture(env_id, lr, lr_ros, num_steps, ros_history, stats, ros_mixture_prob, target_kl):
    subdir = f"history_{ros_history}/mixture_{ros_mixture_prob}/lr_{lr}/lr_ros_{lr_ros}/kl_{target_kl} --eval-freq {EVAL_FREQ[env_id]}"
    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {TIMESTEPS[env_id]}" \
           f" -lr {lr} -ros-lr {lr_ros} -b {ros_history} --num-steps {num_steps} --compute-sampling-error {stats}"

    if target_kl:
        # args += f" --target-kl {target_kl} --ros-target-kl {target_kl}"
        args += f" --ros-target-kl {target_kl}"

    mem = 0.5
    disk = 8

    return mem, disk, args

def ppo(env_id, lr, num_steps, total_timesteps, stats, target_kl):
    # subdir = f"df_{num_steps//2048}/lr_{lr}/kl_{target_kl}"
    subdir = f"df_{num_steps//2048}/lr_{lr}"
    args = f" ppo_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq {EVAL_FREQ[env_id]}" \
           f" -lr {lr} --num-steps {num_steps} --compute-sampling-error {stats}"

    # if target_kl:
    #     args += f" --target-kl {target_kl}"

    mem = 0.5
    disk = 8

    return mem, disk, args

def ppo_buffer(env_id, lr, num_steps, ros_history, stats, target_kl):
    # subdir = f"history_{ros_history}/lr_{lr}/kl_{target_kl}"
    subdir = f"history_{ros_history}/lr_{lr}"
    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {TIMESTEPS[env_id]} --eval-freq {EVAL_FREQ[env_id]}" \
           f" -lr {lr} --ros 0 -b {ros_history} --num-steps {num_steps} --compute-sampling-error {stats}"

    # if target_kl:
    #     args += f" --target-kl {target_kl}"

    mem = 0.5
    disk = 8

    return mem, disk, args

def write_args():


    # env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Ant-v4', 'Walker2d-v4', 'InvertedPendulum-v4', 'InvertedDoublePendulum-v4', ]
    env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Ant-v4', 'Walker2d-v4', 'InvertedDoublePendulum-v4', ]

    env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4', 'Humanoid-v4', 'Swimmer-v4', ]
    # env_ids = ['Humanoid-v4', 'Swimmer-v4', ]

    # env_ids = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']
    # env_ids = ['Hopper-v4']
    # env_ids = ['Ant-v4', 'Humanoid-v4']



    # env_ids = ['Hopper-v4']
    # env_ids = ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4']

    f = open(f"args/discrete.txt", "w")

    num_steps = 2048


    #
    # for env_id in env_ids:
    #     for lr in [1e-3, 1e-4]:
    #         for target_kl in [0.03]:
    #             for ros_history in [2,4,8]:
    #                 args = gen_args(
    #                     device='cpu',
    #                     length="short",
    #                     arg_generator=ppo_buffer,
    #                     env_id=env_id,
    #                     lr=lr,
    #                     target_kl=target_kl,
    #                     num_steps=num_steps,
    #                     ros_history=ros_history,
    #                     stats=0,
    #                 )
    #                 write_to_file(f, args)
    #
    # for env_id in env_ids:
    #     for lr in [1e-3, 1e-4]:
    #         for target_kl in [0.03]:
    #             for df in [1,2,4,8]:
    #                 args = gen_args(
    #                     device='cpu',
    #                     length="short",
    #                     arg_generator=ppo,
    #                     env_id=env_id,
    #                     lr=lr,
    #                     target_kl=target_kl,
    #                     num_steps=int(num_steps*df),
    #                     total_timesteps=int(TIMESTEPS[env_id]*df),
    #                     stats=0,
    #                 )
    #                 write_to_file(f, args)

    # ros_target_kl = 0.05
    # for ros_history in [4]:
    #     for num_steps in [1024, 2048, 4096, 8192]:
    #         for env_id in env_ids:
    #             for lr in [1e-3, 1e-4]:
    #                 for lr_ros in [1e-5]:
    #                     for ros_update_epochs in [16,32]:
    #                         args = gen_args(
    #                             device='cpu',
    #                             length="short",
    #                             arg_generator=ppo_ros,
    #                             env_id=env_id,
    #                             lr=lr,
    #                             lr_ros=lr_ros,
    #                             num_steps=num_steps,
    #                             ros_history=ros_history,
    #                             ros_update_epochs=ros_update_epochs,
    #                             ros_target_kl=ros_target_kl,
    #                             stats=0,
    #                         )
                            # write_to_file(f, args)

    ros_target_kl = 0.05
    for env_id in env_ids:
        f = open(f"args/ros_n256_c0.3_{env_id}.txt", "w")
        for ros_history in [1,2,4]:
            for num_steps in [1024, 2048, 4096, 8192]:
                ros_num_steps = 256
                for lr in [1e-4]:
                    for lr_ros in [1e-3, 1e-4]:
                        for ros_clip_coef in [0.3]:
                            for ros_update_epochs in [2,4,8,16,32]:
                                args = gen_args(
                                    device='cpu',
                                    length="short",
                                    arg_generator=ppo_ros,
                                    env_id=env_id,
                                    lr=lr,
                                    lr_ros=lr_ros,
                                    num_steps=num_steps,
                                    ros_history=ros_history,
                                    ros_num_steps=ros_num_steps,
                                    ros_update_epochs=ros_update_epochs,
                                    ros_target_kl=ros_target_kl,
                                    ros_clip_coef=ros_clip_coef,
                                    stats=0,
                                )
                                write_to_file(f, args)


    # f = open(f"args/discrete_mixture.txt", "w")
    #
    # for env_id in env_ids:
    #     for lr in [1e-3, 1e-4, 1e-5]:
    #         for lr_ros in [1e-4, 1e-5, 1e-6]:
    #             for ros_history in [2,4,8,16]:
    #                 for mixture in [0.5]:
    #                     for num_steps in [256]:
    #                         args = gen_args(
    #                             device='cpu',
    #                             length="short",
    #                             arg_generator=ppo_ros_mixture,
    #                             env_id=env_id,
    #                             lr=lr,
    #                             lr_ros=lr_ros,
    #                             num_steps=num_steps,
    #                             ros_history=ros_history,
    #                             stats=1,
    #                             ros_mixture_prob=mixture
    #                         )
    #                         write_to_file(f, args)

if __name__ == "__main__":

    write_args()


