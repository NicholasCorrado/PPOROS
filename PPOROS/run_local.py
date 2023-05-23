import os

from PPOROS.scripts.utils import TIMESTEPS

if __name__ == "__main__":

    env_ids = ['Swimmer-v4']
    for env_id in env_ids:
        for b in [2, 4]:
            for lr in [1e-3, 1e-4]:
                for s in [1024, 2048, 4096, 8192]:
                    for i in range(10):
                        cmd = f'python ppo_ros_continuous.py -f results_ppo -s b_{b}/s_{s}/lr_{lr} --run-id {i} --seed {i} ' \
                                  f' --env-id {env_id} --total-timesteps {TIMESTEPS[env_id]} --eval-freq 10 --eval-episodes 20 '\
                                  f' -b {b} --num-steps {s} -lr {lr}'
                        print(cmd)
                        os.system(cmd)