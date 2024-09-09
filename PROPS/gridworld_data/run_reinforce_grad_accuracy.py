import os



for i in range(0, 10):
    for oracle in [0]:
        for s in [100]:
            os.system(f'python ../reinforce_discrete.py -f reinforce_tmp -s s_{s} --seed {i} --run-id {i} --linear 1'
                      f' --env-id GridWorld-5x5-v0 --total-timesteps {100*s} --eval-freq {10*s} --eval-episodes 100 --eval 0'
                      f' --props 0 -b {100000} -lr {0} --num-traj {s} '
                      f' --oracle-adaptive {oracle} --gamma 0.99')
#
# for plr in [1e-1]:
#     for i in range(1,5):
#         os.system(f'python ../reinforce_discrete_buf.py -f reinforce_fixed -s plr_{plr} --seed {i} --run-id {i} --linear 1'
#                   f' --env-id GridWorld-5x5-v0 --total-timesteps {5000} --eval-freq {20} --eval-episodes 10000 --anneal-lr 0'
#                   f'  -b {1} --num-steps {100000} -lr {0} --num-traj {1000}'
#                   f' --props 1 -props-lr {plr} --props-num-minibatches 8 --props-update-epochs 16 --props-target-kl 0.1 --props-clip-coef 0.3'
#                   f' --oracle-adaptive 0')




# for i in range(0,50):
#     os.system(f'python ../reinforce_discrete_buf.py -f reinforce_fixed --seed {i} --run-id {i} --linear 1'
#               f' --env-id GridWorld-5x5-v0 --total-timesteps {200000} --eval-freq {5000} --eval-episodes 100 --anneal-lr 0'
#               f'  -b {1} --num-steps {100000} -lr {1e-3} --num-traj {10}'
#               f' --oracle-adaptive 1')