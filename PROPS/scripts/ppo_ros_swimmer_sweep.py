from PPOROS.plotting.condor_no_anneal.ppo import BEST_PPO_PARAMS
from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def write_args():

    env_ids = ['Swimmer-v4']

    for env_id in env_ids:
        f = open(f"args/ros_b24_{env_id}.txt", "w")
        for ros_history in [2]:
            num_steps_list = [1024, 2048, 4096]
            for num_steps in num_steps_list:
                ros_num_steps = 256
                for lr in [1e-3]:
                    for ros_lr in [1e-4, 1e-5]:
                        if num_steps in [1024, 4096] and ros_lr == 1e-4:
                            continue
                        for ros_clip_coef in [0.3]:
                            for ros_update_epochs in [8, 16]:
                                for update_epochs in [10]:
                                    for target_kl in [0.05]:
                                        for ros_target_kl in [0.05]:
                                            args = gen_args(
                                                device='cpu',
                                                length="short",
                                                arg_generator=ppo_ros,
                                                env_id=env_id,
                                                num_steps=num_steps,
                                                lr=lr,
                                                update_epochs=update_epochs,
                                                target_kl=target_kl,
                                                ros_lr=ros_lr,
                                                ros_history=ros_history,
                                                ros_num_steps=ros_num_steps,
                                                ros_update_epochs=ros_update_epochs,
                                                ros_target_kl=ros_target_kl,
                                                ros_clip_coef=ros_clip_coef,
                                                stats=0,
                                                anneal_lr=1
                                            )
                                            write_to_file(f, args)


if __name__ == "__main__":

    write_args()


