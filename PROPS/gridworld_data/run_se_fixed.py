import os

for env_id in ['GridWorld-5x5-v0']:

    for i in range(0, 3):
        for oracle in [0, 1]:
            os.system(f'python ../ppo_props_discrete_clean.py -f tmp_0.99 --seed {i} --run-id {i} --linear 1'
                      f' --env-id {env_id} --env-kwargs rewards:-0.01,0.99,1'
                      f' --total-timesteps {10000} --eval-freq {100} --eval-episodes 100 --eval 0'
                      f' -b {1} -lr {0} --num-steps {10000} '
                      f' --props 0 '
                      f' --oracle-adaptive {oracle}')

    for i in range(0,3):
        os.system(f'python ../ppo_props_discrete_clean.py -f tmp --seed {i} --run-id {i} --linear 1'
                  f' --env-id {env_id} --env-kwargs rewards:-0.01,0.99,1'
                  f' --total-timesteps {200} --eval-freq {100} --eval-episodes 10000 --eval 0'
                  f'  -b {2} -lr {0} --num-steps {100}'
                  f' --props 1 -props-lr {1e-1} --props-num-steps {50} --props-num-minibatches 8 --props-update-epochs 16 --props-target-kl 0.1 --props-clip-coef 0.3'
                  f' --oracle-adaptive 0 --se-init 1')

#
# for plr in [0.05]:
#     for i in range(0,5):
#         os.system(f'python ../reinforce_discrete.py -f reinforce_fixed2 -s plr_{plr} --seed {i} --run-id {i} --linear 1'
#                   f' --env-id GridWorld-5x5-v0 --total-timesteps {10000} --eval-freq {100} --eval-episodes 10000 --eval 0'
#                   f'  -b {1} -lr {0} --num-traj {10000}'
#                   f' --ros 1 -props-lr {plr} --props-num-minibatches 1 --props-update-epochs 1 --props-target-kl 9999999999 --props-clip-coef 999999999'
#                   f' --oracle-adaptive 0 --se-init 1')

# for i in range(0,50):
#     os.system(f'python ../reinforce_discrete_buf.py -f reinforce_fixed --seed {i} --run-id {i} --linear 1'
#               f' --env-id GridWorld-5x5-v0 --total-timesteps {200000} --eval-freq {5000} --eval-episodes 100 --anneal-lr 0'
#               f'  -b {1} --num-steps {100000} -lr {1e-3} --num-traj {10}'
#               f' --oracle-adaptive 1')