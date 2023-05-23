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

def ppo_ros(env_id, total_timesteps, se, num_steps, buffer_history, expert, ros_num_steps, ros_update_epochs, ros_lr, ros_target_kl,
            ros_lambda, ros_clip_coef=0.3):
    if expert:
        subdir = f"expert/b_{buffer_history}/s_{num_steps}/s_{ros_num_steps}/lr_{ros_lr}/l_{ros_lambda}/kl_{ros_target_kl}"
    else:
        subdir = f"random/b_{buffer_history}/s_{num_steps}/s_{ros_num_steps}/lr_{ros_lr}/l_{ros_lambda}/kl_{ros_target_kl}"


    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --ros-num-steps {ros_num_steps} -ros-lr {ros_lr} --ros-update-epochs {ros_update_epochs} --ros-target-kl {ros_target_kl} --ros-anneal-lr 0 " \
           f" --ros-clip-coef {ros_clip_coef}" \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if ros_lambda:
        args += f" --ros-lambda {ros_lambda}"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.8
    disk = 7

    return mem, disk, args

def ppo_buffer(env_id, num_steps, buffer_history, se, total_timesteps, expert):
    if expert:
        subdir = f"expert/b_{buffer_history}"
    else:
        subdir = f"random/b_{buffer_history}"

    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-episodes 0" \
           f" -lr 0 --ros 0 -b {buffer_history} --num-steps {num_steps} --se {se} --eval-freq 1 --se-freq 1" \
           f" --update-epochs 0 --ros-anneal-lr 0" \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.8
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


    env_ids = ['Ant-v4', 'Humanoid-v4', 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', ]

    f = open(f"data/se.txt", "w")


    env_ids = ['Swimmer-v4', 'Humanoid-v4'  ]

    for expert in [1]:
        for buffer_history in [16]:
            num_collects = buffer_history * 2

            for env_id in env_ids:
                for num_steps in [1024, 2048, 4096, 8192]:
                    for ros_lambda in [0.01, 0.1, 0.3]:
                        for ros_lr in [1e-3, 1e-4]:
                            for ros_num_steps in [num_steps]:
                                for ros_update_epochs in [16]:
                                    for ros_target_kl in [0.05]:

                                        if ros_num_steps > num_steps: continue
                                        args = gen_args(
                                            device='cuda',
                                            length="short",
                                            arg_generator=ppo_ros,
                                            env_id=env_id,
                                            ros_lr=ros_lr,
                                            ros_update_epochs=ros_update_epochs,
                                            num_steps=num_steps,
                                            buffer_history=buffer_history,
                                            se=1,
                                            total_timesteps=num_steps*num_collects,
                                            expert=expert,
                                            ros_num_steps=ros_num_steps,
                                            ros_target_kl=ros_target_kl,
                                            ros_lambda=ros_lambda,
                                        )
                                        write_to_file(f, args)


if __name__ == "__main__":

    write_args()


