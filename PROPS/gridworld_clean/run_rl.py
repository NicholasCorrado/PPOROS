import os

# for b in [1]:
#     for i in range(0,20):
        # 5x5
        # df = 1
        # b = 1
        # lr = 1e-3
        # num_traj = 4



for s in [10]:
    total_timesteps = 5000 * s//10
    lr = 1e-2
    eval_freq = 100


    for i in range(0, 100):
        # os.system(f'python ../ppo_og.py '
        #           f' --output-rootdir rl '
        #           f' --output_subdir lr_{lr}/s_{s}'
        #           f' --seed {i} --run-id {i} '
        #           f' --env-id GridWorld1D-10-v0 '
        #           f' --total-timesteps {total_timesteps} --eval-freq {eval_freq} --eval-episodes 100'
        #           f' --learning-rate {lr} --num-steps {s} '
        #           f' --linear {1}')

        os.system(f'python ../ppo_og.py --output-rootdir rl '
                  f' --output_subdir lr_{lr}/s_{s}'
                  f' --seed {i} --run-id {i} --linear {1}'
                  f' --env-id GridWorld1D-10-v0 --total-timesteps {total_timesteps} '
                  f' --eval-freq {eval_freq} --eval-episodes 100'
                  f' --learning-rate {lr} --num-steps {s} '
                  f' --sampling-algo props'
                  f' --props-num-steps {1}'
                  f' --props-learning-rate {1e-1}'
                  f' --props-update-epochs {8}'
                  f' --props-num-minibatches {1}'
                  f' --props-target-kl {0.5}'
                  f' --props-lambda 0'
                  )

    # os.system(f'python ../reinforce_discrete.py -f rl3 --seed {i} --run-id {i} --linear 1'
    #           f' --env-id GridWorld-5x5-v0 --total-timesteps {300000} --eval-freq {10000} --eval-episodes 100'
    #           f' --props 0 -b {1} -lr {3e-3} --num-traj {100} '
    #           f' --props 1 -props-lr {1e-1} --props-num-traj 5 --props-num-minibatches 8 --props-update-epochs 4 --props-target-kl 0.3 --props-clip-coef 0.3'
    #           f' --se 0 --se-freq 1 --track 1'
    #           f' --oracle-adaptive 0 ')
#
# for plr in [1e-1]:
#     for i in range(1,5):
#         os.system(f'python ../reinforce_discrete_buf.py -f reinforce_fixed -s plr_{plr} --seed {i} --run-id {i} --linear 1'
#                   f' --env-id GridWorld-5x5-v0 --total-timesteps {5000} --eval-freq {20} --eval-episodes 10000 --anneal-lr 0'
#                   f'  -b {1} --num-steps {100000} -lr {0} --num-traj {1000}'
#                   f' --props 1 -props-lr {plr} --props-num-minibatches 8 --props-update-epochs 16 --props-target-kl 0.1 --props-clip-coef 0.3'
#                   f' --oracle-adaptive 0')
