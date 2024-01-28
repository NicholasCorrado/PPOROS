import os



for b in [1]:
    for i in range(10):
        # GOOD
        # os.system(f'python ../ppo_props_discrete.py -f results -s b_{b}'
        #           f' --env-id TwoStep-v0 --total-timesteps 40000 --eval-freq 10 --eval-episodes 100'
        #           f'  -b {b}  --num-steps 200 -lr 0.5 --update-epochs 1 --num-minibatches 1 '
        #           f' --props 1 --props-target-kl 1 --props-clip-coef 0.3 --props-lambda 0 --props-num-steps 5 --props-update-epochs 16 -props-lr 0.5 --props-num-minibatches 1'
        #           f' --se 0 --se-freq 20 --se-epochs 100 ')

        os.system(f'python ppo_props_discrete.py -f results -s b_{b}'
                  f' --env-id TwoStep-v0 --total-timesteps 40000 --eval-freq 10 --eval-episodes 100'
                  f'  -b {b}  --num-steps 200 -lr 0.5 --update-epochs 1 --num-minibatches 1 '
                  f' --props 1 --props-target-kl 1 --props-clip-coef 0.3 --props-lambda 0 --props-num-steps 5 --props-update-epochs 16 -props-lr 0.5 --props-num-minibatches 1'
                  f' --se 1 --se-freq 1 ')
