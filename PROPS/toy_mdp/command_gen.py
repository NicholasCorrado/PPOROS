for lr in [1]:
    for b in [1, 2]:
        for props in [1]:
                # GOOD
                cmd = f'0.4,8,ppo_props_discrete.py -f results -s b_{b}/lr_{lr}' \
                          f' --env-id TwoStep-v0 --total-timesteps 40000 --eval-freq 2 --eval-episodes 100'\
                          f'  -b {b}  --num-steps 200 -lr 0.5 --update-epochs 1 --num-minibatches 1 '\
                          f' --props {props} --props-target-kl 1 --props-clip-coef 0.3 --props-lambda 0 --props-num-steps 5 --props-update-epochs 16 -props-lr {lr} --props-num-minibatches 1'\
                          f' --se 0 --se-freq 20 --se-epochs 100'

                print(cmd.replace(' ', '*'))