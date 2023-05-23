import os

if __name__ == "__main__":


    """
    0.5,8, ppo_continuous.py --env-id Swimmer-v4 -s num_steps_2048/lr_0.001/kl_None --total-timesteps 2000000 --eval-freq 10 -lr 0.001 --num-steps 2048 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1
0.5,8, ppo_continuous.py --env-id Hopper-v4 -s num_steps_2048/lr_0.0005/kl_None --total-timesteps 2000000 --eval-freq 10 -lr 0.0005 --num-steps 2048 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1
0.5,8, ppo_continuous.py --env-id Walker2d-v4 -s num_steps_8192/lr_0.0005/kl_None --total-timesteps 5000000 --eval-freq 10 -lr 0.0005 --num-steps 8192 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1
0.5,8, ppo_continuous.py --env-id HalfCheetah-v4 -s num_steps_1024/lr_0.0001/kl_None --total-timesteps 5000000 --eval-freq 10 -lr 0.0001 --num-steps 1024 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1
0.5,8, ppo_continuous.py --env-id Ant-v4 -s num_steps_1024/lr_0.0001/kl_None --total-timesteps 6000000 --eval-freq 10 -lr 0.0001 --num-steps 1024 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1
0.5,8, ppo_continuous.py --env-id Humanoid-v4 -s num_steps_8192/lr_0.0001/kl_None --total-timesteps 6000000 --eval-freq 10 -lr 0.0001 --num-steps 8192 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1
    
    """

    os.system(f'python ppo_continuous.py --env-id Hopper-v4 -f local_experts -s num_steps_2048/lr_0.0001/kl_None --total-timesteps 2000000 --eval-freq 10 -lr 0.0001 --num-steps 2048 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1')
    os.system(f'python ppo_continuous.py --env-id Swimmer-v4 -f local_experts --total-timesteps 2000000 --eval-freq 10 -lr 0.001 --num-steps 2048 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1')
    os.system(f'python ppo_continuous.py --env-id Ant-v4 -f local_experts -s num_steps_1024/lr_0.0001/kl_None --total-timesteps 6000000 --eval-freq 10 -lr 0.0001 --num-steps 1024 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1')
    os.system(f'python ppo_continuous.py --env-id Humanoid-v4 -f local_experts -s num_steps_8192/lr_0.0001/kl_None --total-timesteps 6000000 --eval-freq 10 -lr 0.0001 --num-steps 8192 --num-minibatches 32  --compute-sampling-error 0 --save-policy 1')
