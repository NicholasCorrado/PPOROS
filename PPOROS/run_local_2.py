import os

if __name__ == "__main__":

    for i in range(10):
        os.system(f'python ppo_ros_discrete.py -f results_local --env-id CartPole-v1 --learning-rate 0 '
                  f' --ros 0 -b 5 --total-timesteps {256*5} --eval-freq 1 --compute-sampling-error 1')

