TIMESTEPS = {
    # 'Pendulum-v1': int(5e6),
    # 'InvertedPendulum-v4': int(100e3),
    # 'InvertedDoublePendulum-v4': int(3e6),
    # 'Reacher-v4': int(10e6),
    'HalfCheetah-v4': int(10e6),
    'Hopper-v4': int(10e6),
    'Ant-v4': int(10e6),
    'Walker2d-v4': int(10e6),
    'Humanoid-v4': int(10e6),
    'Swimmer-v4': int(10e6),
}
if __name__ == "__main__":

    commands = ""
    mem = 0.8
    disk = 2


    # env_ids = [
    #     'GridWorld-5x5-v0',
    #     'GridWorld-10x10-v0',
    #     'GridWorld-20x20-v0',
    # ]
    for l in [10, 20]:
        env_id = f'GridWorld-{l}x{l}-v0'
        # for df in [1, 2, 4, 8, 16, 32]:
        for b in [1, 2, 4, 8, 16, 32]:
            for lr in [1e-3]:
                for s in [2*l, 4*l, 8*l, 16*l, 32*l, 64*l]:
                    # b = 1
                    se = 0
                    props = 0
                    ps = s
                    plr = 0
                    pkl = 0
                    commands += f'{mem},{disk},ppo_props_discrete.py --env-id {env_id} ' \
                                f' -s lr_{lr}/s_{s}/b_{b}' \
                                f' --total-timesteps {4000*s} --eval-freq {s * 100} --eval-episodes 100 ' \
                                f' --actor-critic 1 -b {b} --num-steps {s} -lr {lr} --anneal-lr 0 --linear 1'\
                                f' --props {props} --props-num-steps {ps} -props-lr {plr} --props-target-kl {pkl} ' \
                                f' --props-clip-coef 0.3 --props-lambda 0.1 --props-update-epochs 4  --props-num-minibatches 4' \
                                f' --se 0 --se-freq 1 --track 0\n'


    commands = commands[:-1]
    commands = commands.replace(' ', '*')
    print(commands)