import os



for b in [1, 2]:
    for props in [0, 1]:
        for i in range(7):
            # GOOD
            # os.system(f'python ../ppo_props_discrete.py -f results -s b_{b}'
            #           f' --env-id TwoStep-v0 --total-timesteps 40000 --eval-freq 10 --eval-episodes 100'
            #           f'  -b {b}  --num-steps 200 -lr 0.5 --update-epochs 1 --num-minibatches 1 '
            #           f' --props 1 --props-target-kl 1 --props-clip-coef 0.3 --props-lambda 0 --props-num-steps 5 --props-update-epochs 16 -props-lr 0.5 --props-num-minibatches 1'
            #           f' --se 0 --se-freq 20 --se-epochs 100 ')

            os.system(f'python ppo_props_discrete.py -f results_b1 -s b_{b}'
                      f' --env-id GridWorld-v0 --total-timesteps 60000 --eval-freq 10 --eval-episodes 100 --anneal-lr 0'
                      f'  -b {b}  --num-steps 300 -lr 0.01 --update-epochs 1 --num-minibatches 1 '
                      f' --props {props} --props-target-kl 1 --props-clip-coef 0.3 --props-lambda 0 --props-num-steps 30 --props-update-epochs 16 -props-lr 0.005 --props-num-minibatches 1'
                      f' --se 0 --se-freq 1 ')
