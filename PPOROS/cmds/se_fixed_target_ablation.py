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
    # default_args = f"{mem},{disk},"
    default_args = f""
    args = default_args + other_args

    return args

def ppo_props_no_clip(env_id, total_timesteps, se, num_steps, buffer_history, expert,  props_lr,props_lambda):
    if expert:
        subdir = f"expert/no_clip"
    else:
        subdir = f"random/no_clip"

    # subdir = f"expert/b_{buffer_history}"


    args = f"python ppo_props_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --props-num-steps 256 --props-update-epochs 16 --props-target-kl 0.05 --props-clip-coef 999999999 --props-lambda {props_lambda}" \
           f" -props-lr {props_lr} --props-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def ppo_props_no_lambda(env_id, total_timesteps, se, num_steps, buffer_history, expert,  props_lr,props_lambda):
    if expert:
        subdir = f"expert/no_lambda"
    else:
        subdir = f"random/no_lambda"

    # subdir = f"expert/b_{buffer_history}"


    args = f"python ppo_props_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --props-num-steps 256 --props-update-epochs 16 --props-target-kl 0.05 --props-clip-coef 0.3 --props-lambda 0" \
           f" -props-lr {props_lr} --props-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def ppo_props_no_target(env_id, total_timesteps, se, num_steps, buffer_history, expert,  props_lr,props_lambda):
    if expert:
        subdir = f"expert/no_target"
    else:
        subdir = f"random/no_target"

    # subdir = f"expert/b_{buffer_history}"


    args = f"python ppo_props_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --props-num-steps 256 --props-update-epochs 16 --props-target-kl 999999999 --props-clip-coef 0.3 --props-lambda {props_lambda}" \
           f" -props-lr {props_lr} --props-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def ppo_props_none(env_id, total_timesteps, se, num_steps, buffer_history, expert,  props_lr,props_lambda):
    if expert:
        subdir = f"expert/none"
    else:
        subdir = f"random/none"

    # subdir = f"expert/b_{buffer_history}"


    args = f"python ppo_props_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --props-num-steps 256 --props-update-epochs 16 --props-target-kl 999999999 --props-clip-coef 999999999 --props-lambda 0" \
           f" -props-lr {props_lr} --props-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def ppo_props_clip_lambda(env_id, total_timesteps, se, num_steps, buffer_history, expert,  props_lr,props_lambda):
    if expert:
        subdir = f"expert/b_{buffer_history}/no_clip_lambda"
    else:
        subdir = f"random/b_{buffer_history}/no_clip_lambda"

    # subdir = f"expert/b_{buffer_history}"


    args = f"python ppo_props_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} -lr 0 --update-epochs 0 --anneal-lr 0" \
           f" --props-num-steps 256 --props-update-epochs 16 --props-target-kl 0.05 --props-clip-coef 999999999 --props-lambda 0" \
           f" -props-lr {props_lr} --props-anneal-lr 0 " \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 0.7
    disk = 7

    return mem, disk, args

def write_args():


    env_ids = ['Ant-v4', 'Humanoid-v4', 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', ]

    f = open(f"commands/se_fixed_target_props_ablations.txt", "w")
    for expert in [1]:
        for buffer_history in [16]:
            num_collects = buffer_history * 2
            for env_id in env_ids:
                num_steps = 1024

                if env_id in ['Swimmer-v4', 'Ant-v4', 'Humanoid-v4']:
                    props_lr = 1e-4
                else:
                    props_lr = 1e-3

                if env_id in ['Ant-v4', 'Humanoid-v4']:
                    props_lambda = 0.3
                else:
                    props_lambda = 0.1

                # for func in [ppo_props_no_clip, ppo_props_no_lambda, ppo_props_clip_lambda, ppo_props_none, ppo_props_no_target]:
                for func in [ppo_props_no_clip, ppo_props_no_lambda, ppo_props_clip_lambda]:

                    args = gen_args(
                        device='cuda',
                        length="short",
                        arg_generator=func,
                        env_id=env_id,
                        props_lr=props_lr,
                        num_steps=num_steps,
                        buffer_history=buffer_history,
                        se=1,
                        total_timesteps=num_steps*num_collects,
                        expert=expert,
                        props_lambda=props_lambda,
                    )
                    write_to_file(f, args)

if __name__ == "__main__":

    write_args()


