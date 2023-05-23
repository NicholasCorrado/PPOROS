import os

if __name__ == "__main__":

    num_steps = 1024
    b = 16
    expert = True

    ros_epochs = 32
    ros_lr = 1e-4
    env_id = 'Hopper-v4'

    env_ids = ['Hopper-v4', 'Swimmer-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4']

    for env_id in env_ids:
        for i in range(5):
            for ros_lr in [1e-3, 1e-4]:


                command = f'python ppo_ros_continuous.py -f results_se -s lr_{ros_lr} --seed {i} --run-id {i} ' \
                          f' --env-id {env_id} --total-timesteps {num_steps * b*2} --eval-freq 100 --eval-episodes 0' \
                          f' -b {b} -lr 0 --update-epochs 0 --num-steps {num_steps} ' \
                          f' --ros {1} --ros-vanilla 1 --ros-num-steps {1} -ros-lr {ros_lr} --ros-target-kl 999999999 --ros-lambda 0 --ros-update-epochs 1 --ros-num-minibatches 1 --ros-clip-coef 999999999' \
                          f' --se 1 --se-freq 1 --se-epochs 1000 --se-lr 1e-3'

                if expert:
                    command += f' --policy-path policies/{env_id}/best_model.zip --normalization-dir policies/{env_id}'

                print(command)
                os.system(command)


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

