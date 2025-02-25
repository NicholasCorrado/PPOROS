import os

output_dir = 'se_fixed_new'
total_timesteps = 1000
eval_freq = 20
props_lr = 1e-1
ros_lr = 1e4
props_num_steps = 1

# for env_id in ['GridWorld-5x5-v0']:
for env_id in ['GridWorld1D-10-v0']:

    for i in range(0, 3):
        for oracle in [0, 1]:
            os.system(f'python ../ppo_props_discrete_clean.py -f {output_dir} --seed {i} --run-id {i} --linear 1'
                      f' --env-id {env_id} --env-kwargs rewards:-0.01,0.5,1'
                      f' --total-timesteps {total_timesteps} --eval-freq {eval_freq} --eval-episodes 100 --eval 0'
                      f' -b {1} -lr {0} --num-steps {total_timesteps} '
                      f' --props 0 '
                      f' --oracle-adaptive {oracle}')

    # for i in range(0, 10):
    #     os.system(f'python ../ppo_props_discrete_clean.py -f se_fixed_final_test --seed {i} --run-id {i} --linear 1'
    #               f' --env-id {env_id} --env-kwargs rewards:-0.01,0.5,1'
    #               f' --total-timesteps {total_timesteps} --eval-freq {eval_freq} --eval-episodes 100 --eval 0'
    #               f'  -b {2} -lr {0} --num-steps {total_timesteps}'
    #               f' --props 1 -props-lr {props_lr} --props-num-steps {props_num_steps} --props-num-minibatches 8 --props-update-epochs 32 --props-target-kl 0.5 --props-clip-coef 1'
    #               f' --oracle-adaptive 0 --se-init 1')

    # for i in range(0, 10):
    #     os.system(f'python ../ppo_props_discrete_clean.py -f se_fixed_final --seed {i} --run-id {i} --linear 1'
    #               f' --env-id {env_id} --env-kwargs rewards:-0.01,0.5,1'
    #               f' --total-timesteps {total_timesteps} --eval-freq {eval_freq} --eval-episodes 100 --eval 0'
    #               f'  -b {1} -lr {0} --num-steps {total_timesteps}'
    #               f' --ros 1 -props-lr {ros_lr}'
    #               f' --oracle-adaptive 0 --se-init 1')

#
# for plr in [0.05]:
#     for i in range(0,5):
#         os.system(f'python ../reinforce_discrete.py -f reinforce_fixed2 -s plr_{plr} --seed {i} --run-id {i} --linear 1'
#                   f' --env-id GridWorld-5x5-v0 --total-timesteps {10000} --eval-freq {100} --eval-episodes 10000 --eval 0'
#                   f'  -b {1} -lr {0} --num-traj {10000}'
#                   f' --ros 1 -props-lr {plr} --props-num-minibatches 1 --props-update-epochs 1 --props-target-kl 9999999999 --props-clip-coef 999999999'
#                   f' --oracle-adaptive 0 --se-init 1')

# for i in range(0,50):
#     os.system(f'python ../reinforce_discrete_buf.py -f se_fixed_ros --seed {i} --run-id {i} --linear 1'
#               f' --env-id GridWorld-5x5-v0 --total-timesteps {total_timesteps} --eval-freq {eval_freq} --eval-episodes 100 --anneal-lr 0'
#               f'  -b {1} --num-steps {100000} -lr {1e-3}'
#               f' --oracle-adaptive 1')