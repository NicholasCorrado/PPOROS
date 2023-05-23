import os

if __name__ == "__main__":

    num_steps = 1024
    b = 16
    expert = True

    ros_lr = 1e-3
    env_id = 'Hopper-v4'

    for i in range(5):
        for ros in [1]:
            command = f'python ppo_ros_continuous.py -f results_se --seed {i} --run-id {i} ' \
                      f' --env-id {env_id} --total-timesteps {num_steps * b} --eval-freq 100 --eval-episodes 0' \
                      f' -b {b} -lr 0 --update-epochs 0 --num-steps {num_steps} ' \
                      f' --ros {ros} --ros-num-steps {256} -ros-lr {ros_lr} --ros-target-kl 0.05 --ros-lambda 0.1 --ros-update-epochs 16' \
                      f' --se 1 --se-freq 1 --se-epochs 1000 --se-lr 1e-3'

            if expert:
                command += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'

            print(command)
            os.system(command)
"""
diff [9.2067289352417, 0.5680875778198242, 0.6710300445556641, -0.47551393508911133, 0.028372198343276978, 0.0011079907417297363]
ref [6.141351699829102, 2.833974838256836, 1.0757187604904175, 1.1628758907318115, 0.4439198076725006, 0.33983907103538513]
[15.3480806350708, 3.40206241607666, 1.7467488050460815, 0.6873619556427002, 0.4722920060157776, 0.34094706177711487]
"""

    # for i in range(5):
    #     for ros in [1]:
    #         command = f'python ppo_ros_continuous.py -f results_se --seed {i} --run-id {i} ' \
    #                   f' --env-id {env_id} --total-timesteps {num_steps * b} --eval-freq 100 --eval-episodes 0' \
    #                   f' -b {b} -lr 0 --update-epochs 0 --num-steps {num_steps} ' \
    #                   f' --ros {ros} --ros-num-steps {num_steps} -ros-lr {ros_lr} --ros-target-kl 0.05 --ros-lambda 0.1 --ros-update-epochs 16' \
    #                   f' --se 1 --se-freq 1 --se-epochs 1000 --se-lr 1e-3'
    #
    #         if expert:
    #             command += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    #
    #         print(command)
    #         os.system(command)

    # env_ids = ['Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4']
    #
    # for env_id in env_ids:
    #
    #     for i in range(5):
    #         for ros in [0, 1]:
    #             command = f'python ppo_ros_continuous.py -f results_se --seed {i} --run-id {i} ' \
    #                       f' --env-id {env_id} --total-timesteps {num_steps*b} --eval-freq 100 --eval-episodes 0' \
    #                       f' -lr 0 --update-epochs 0 --num-steps {num_steps} ' \
    #                       f' --ros {ros} -b {b} -ros-lr {ros_lr} --ros-target-kl 0.05 --ros-lambda 0.1 --ros-update-epochs 16' \
    #                       f' --se 1 --se-freq 1 --se-epochs 1000 --se-lr 1e-3'
    #
    #             if expert:
    #                 command += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'
    #
    #             print(command)
    #             os.system(command)

