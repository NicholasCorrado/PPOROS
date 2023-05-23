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
    default_args = ""
    args = default_args + other_args

    return args

def ppo_props(env_id, total_timesteps, se, num_steps, buffer_history, expert, props_num_steps, props_update_epochs, props_lr, props_target_kl,
            props_lambda, props_clip_coef=0.3):
    if expert:
        subdir = f"expert"
    else:
        subdir = f"random"

    args = f"python ppo_props_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-freq 1 --eval-episodes 0" \
           f" -b {buffer_history} --num-steps {num_steps} --update-epochs 0 " \
           f" --props-num-steps {props_num_steps} -props-lr {props_lr} --props-target-kl {props_target_kl}" \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if props_lambda:
        args += f" --props-lambda {props_lambda}"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 2
    disk = 7

    return mem, disk, args

def ppo_buffer(env_id, num_steps, buffer_history, se, total_timesteps, expert):
    if expert:
        subdir = f"expert"
    else:
        subdir = f"random"

    args = f"python ppo_props_continuous.py --env-id {env_id} -s {subdir} --total-timesteps {total_timesteps} --eval-episodes 0" \
           f" --props 0 -b {buffer_history} --num-steps {num_steps} --se {se} --eval-freq 1 --update-epochs 0" \
           f" --se {se} --se-freq 1 --se-lr 1e-3 --se-epochs 1000"

    if expert:
        args += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    mem = 2
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



    f = open(f"commands/se_fixed_target_on_policy.txt", "w")

    env_ids = ['Ant-v4', 'Humanoid-v4', 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', ]
    for expert in [0, 1]:
        for env_id in env_ids:
            args = gen_args(
                device='cpu',
                length="short",
                arg_generator=ppo_buffer,
                env_id=env_id,
                num_steps=1024,
                buffer_history=32,
                se=1,
                total_timesteps=1024 * 64,
                expert=expert,
            )
            write_to_file(f, args)

    f = open(f"commands/se_fixed_target_props.txt", "w")

    for expert in [0,1]:
        for env_id in env_ids:

            if env_id in ['Ant-v4', 'Humanoid-v4']:
                props_lambda = 0.3
                props_lr = 1e-4
            elif env_id in ['Swimmer-v4']:
                props_lambda = 0.1
                props_lr = 1e-4
            else:
                props_lambda = 0.1
                props_lr = 1e-3

            args = gen_args(
                device='cuda',
                length="short",
                arg_generator=ppo_props,
                env_id=env_id,
                props_lr=props_lr,
                props_update_epochs=16,
                num_steps=1024,
                buffer_history=16,
                se=1,
                total_timesteps=1024*32,
                expert=expert,
                props_num_steps=256,
                props_target_kl=0.05,
                props_lambda=props_lambda,
            )
            write_to_file(f, args)

if __name__ == "__main__":

    write_args()


