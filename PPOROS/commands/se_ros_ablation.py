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

def ppo_ros_no_clip(env_id, total_timesteps, se, num_steps, buffer_history, expert,  ros_lr,ros_lambda):
    if expert:
        subdir = f"expert/b_{buffer_history}/no_clip"
    else:
        subdir = f"random/b_{buffer_history}/no_clip"

    # subdir = f"expert/b_{buffer_history}"


    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --ros-num-steps 256 --ros-update-epochs 16 --ros-target-kl 0.05 --ros-clip-coef 999999999 --ros-lambda {ros_lambda}" \
           f" -ros-lr {ros_lr} --ros-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def ppo_ros_no_lambda(env_id, total_timesteps, se, num_steps, buffer_history, expert,  ros_lr,ros_lambda):
    if expert:
        subdir = f"expert/b_{buffer_history}/no_lambda"
    else:
        subdir = f"random/b_{buffer_history}/no_lambda"

    # subdir = f"expert/b_{buffer_history}"


    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --ros-num-steps 256 --ros-update-epochs 16 --ros-target-kl 0.05 --ros-clip-coef 0.3 --ros-lambda 0" \
           f" -ros-lr {ros_lr} --ros-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def ppo_ros_no_target(env_id, total_timesteps, se, num_steps, buffer_history, expert,  ros_lr,ros_lambda):
    if expert:
        subdir = f"expert/b_{buffer_history}/no_target"
    else:
        subdir = f"random/b_{buffer_history}/no_target"

    # subdir = f"expert/b_{buffer_history}"


    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --ros-num-steps 256 --ros-update-epochs 16 --ros-target-kl 999999999 --ros-clip-coef 0.3 --ros-lambda {ros_lambda}" \
           f" -ros-lr {ros_lr} --ros-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def ppo_ros_none(env_id, total_timesteps, se, num_steps, buffer_history, expert,  ros_lr,ros_lambda):
    if expert:
        subdir = f"expert/b_{buffer_history}/none"
    else:
        subdir = f"random/b_{buffer_history}/none"

    # subdir = f"expert/b_{buffer_history}"


    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --ros-num-steps 256 --ros-update-epochs 16 --ros-target-kl 999999999 --ros-clip-coef 999999999 --ros-lambda 0" \
           f" -ros-lr {ros_lr} --ros-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def ppo_ros_clip_lambda(env_id, total_timesteps, se, num_steps, buffer_history, expert,  ros_lr,ros_lambda):
    if expert:
        subdir = f"expert/b_{buffer_history}/no_clip_lambda"
    else:
        subdir = f"random/b_{buffer_history}/no_clip_lambda"

    # subdir = f"expert/b_{buffer_history}"


    args = f" ppo_ros_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --ros-num-steps 256 --ros-update-epochs 16 --ros-target-kl 0.05 --ros-clip-coef 999999999 --ros-lambda 0" \
           f" -ros-lr {ros_lr} --ros-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def write_args():


    env_ids = ['Ant-v4', 'Humanoid-v4', 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', ]

    f = open(f"data/se.txt", "w")
    for expert in [1]:
        for buffer_history in [16]:
            num_collects = buffer_history * 2
            for env_id in env_ids:
                num_steps = 1024

                if env_id in ['Swimmer-v4', 'Ant-v4', 'Humanoid-v4']:
                    ros_lr = 1e-4
                else:
                    ros_lr = 1e-3

                if env_id in ['Ant-v4', 'Humanoid-v4']:
                    ros_lambda = 0.3
                else:
                    ros_lambda = 0.1

                # for func in [ppo_ros_no_clip, ppo_ros_no_lambda, ppo_ros_clip_lambda, ppo_ros_none, ppo_ros_no_target]:
                for func in [ppo_ros_no_clip, ppo_ros_no_lambda, ppo_ros_clip_lambda]:

                    args = gen_args(
                        device='cuda',
                        length="short",
                        arg_generator=func,
                        env_id=env_id,
                        ros_lr=ros_lr,
                        num_steps=num_steps,
                        buffer_history=buffer_history,
                        se=1,
                        total_timesteps=num_steps*num_collects,
                        expert=expert,
                        ros_lambda=ros_lambda,
                    )
                    write_to_file(f, args)

if __name__ == "__main__":

    write_args()


