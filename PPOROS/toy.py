import os

if __name__ == "__main__":


    for k in range(5):
        num_steps = 256*2**k
    for num_steps in [8192, 512]:
        for i in range(10, 20):
            os.system(f'python ppo_continuous.py -f results_toy -s s_{num_steps} --run-id {i} --seed {i} '
                      f' --env-id Goal1D-v0 --total-timesteps {int(200e3)} --eval-freq 1 --eval-episodes 1 '
                      f' -lr 1e-3 --num-steps {num_steps}')

