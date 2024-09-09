import os



for b in [1, 32]:
    for props in [0]:
        for i in range(5):
            # 5x5
            df = 1
            s = 10*df
            # b = 1
            lr = 1e-3
            os.system(f'python ../ppo_props_discrete.py -f results_linear2 -s lr_{lr}/s_{s}/b_{b} --seed {i} --run-id {i} --actor-critic 1 --linear 1'
                      f' --env-id GridWorld-5x5-v0 --total-timesteps {5000*s} --eval-freq {s*500} --eval-episodes 10 --anneal-lr 0'
                      f'  -b {b} --num-steps {s} -lr {lr}'
                      f' --props {props} --props-target-kl 0.1 --props-clip-coef 0.3 --props-lambda 0.1 --props-num-steps 10 --props-update-epochs 4 -props-lr 0.001 --props-num-minibatches 4'
                      f' --se 0 --se-freq 1 --track 0')
