import os

# for b in [1]:
#     for props in [0]:
#         for i in range(0,50):
#             # 5x5
#             df = 1
#             # b = 1
#             lr = 1e-3
#             num_traj = 4
#             os.system(f'python ../reinforce_discrete_buf.py -f reinforce -s lr_{lr}/s_{num_traj}/b_{b} --seed {i} --run-id {i} --linear 1'
#                       f' --env-id GridWorld-5x5-v0 --total-timesteps {200000} --eval-freq {5000} --eval-episodes 100 --anneal-lr 0'
#                       f'  -b {b} --num-steps {1000} -lr {lr} --num-traj {num_traj}'
#                       f' --props {props} --props-target-kl 0.1 --props-clip-coef 0.3 --props-lambda 0.1 --props-num-steps 10 --props-update-epochs 4 -props-lr 0.001 --props-num-minibatches 4'
#                       f' --se 0 --se-freq 1 --track 0'
#                       f' --oracle-adaptive 1')
#

for lr in [1e-1]:
    for s in [100]:
        for i in range(10):
            os.system(f'python ../reinforce_discrete_buf.py --seed {i} --run-id {i} --linear 1'
                      f' -f reinforce_sillybandit -s lr_{lr}/s_{s}'
                      f' --env-id SillyBandit-v0 '
                      f' --total-timesteps {10000*s} --eval-freq {500*s} --eval-episodes 1 --anneal-lr 0'
                      f' --props 0 -b {1} --num-steps {100000*s} -lr {lr} --num-traj {s} ')

# for plr in [1e-1]:
#     for i in range(1,5):
#         os.system(f'python ../reinforce_discrete_buf.py -f reinforce_fixed -s plr_{plr} --seed {i} --run-id {i} --linear 1'
#                   f' --env-id GridWorld-5x5-v0 --total-timesteps {5000} --eval-freq {20} --eval-episodes 10000 --anneal-lr 0'
#                   f'  -b {1} --num-steps {100000} -lr {0} --num-traj {1000}'
#                   f' --props 1 -props-lr {plr} --props-num-minibatches 8 --props-update-epochs 16 --props-target-kl 0.1 --props-clip-coef 0.3'
#                   f' --oracle-adaptive 0')
