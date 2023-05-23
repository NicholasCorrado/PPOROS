from PPOROS.hyperparams.ppo_ros import PPOROS_PARAMS
from PPOROS.scripts.utils import ppo_ros, gen_args, write_to_file


def write_args():

    #
    # params_list = [
    #     {
    #         'buffer_size': 2,
    #         'num_steps': 1024,
    #         'ros_num_steps': 256,
    #         'lr': 1e-3,
    #         'ros_lr': 1e-4,
    #         'update_epochs': 10,
    #         'ros_update_epochs': 16,
    #         'target_kl': 0.05,
    #         'ros_target_kl': 0.05,
    #         'clip_coef': 0.2,
    #         'ros_clip_coef': 0.3,
    #     },
    #     {
    #         'buffer_size': 2,
    #         'num_steps': 2048,
    #         'ros_num_steps': 256,
    #         'lr': 1e-4,
    #         'ros_lr': 1e-3,
    #         'update_epochs': 10,
    #         'ros_update_epochs': 16,
    #         'target_kl': 0.05,
    #         'ros_target_kl': 0.05,
    #         'clip_coef': 0.2,
    #         'ros_clip_coef': 0.3,
    #     },
    #     {
    #         'buffer_size': 2,
    #         'num_steps': 2048,
    #         'ros_num_steps': 256,
    #         'lr': 1e-4,
    #         'ros_lr': 1e-3,
    #         'update_epochs': 10,
    #         'ros_update_epochs': 4,
    #         'target_kl': 0.05,
    #         'ros_target_kl': 0.05,
    #         'clip_coef': 0.2,
    #         'ros_clip_coef': 0.3,
    #     },
    #     {
    #         'buffer_size': 2,
    #         'num_steps': 2048,
    #         'ros_num_steps': 256,
    #         'lr': 1e-4,
    #         'ros_lr': 1e-3,
    #         'update_epochs': 10,
    #         'ros_update_epochs': 4,
    #         'target_kl': 0.1,
    #         'ros_target_kl': 0.05,
    #         'clip_coef': 0.2,
    #         'ros_clip_coef': 0.3,
    #     },
    #     {
    #         'buffer_size': 2,
    #         'num_steps': 2048,
    #         'ros_num_steps': 256,
    #         'lr': 1e-4,
    #         'ros_lr': 1e-3,
    #         'update_epochs': 10,
    #         'ros_update_epochs': 16,
    #         'target_kl': 0.1,
    #         'ros_target_kl': 0.05,
    #         'clip_coef': 0.2,
    #         'ros_clip_coef': 0.3,
    #     },
    # ]


    params_list = [
        {
            'buffer_size': 2,
            'num_steps': 2048,
            'ros_num_steps': 1024,
            'lr': 1e-4,
            'ros_lr': 1e-3,
            'update_epochs': 10,
            'ros_update_epochs': 4,
            'target_kl': 0.03,
            'ros_target_kl': 0.05,
            'ros_clip_coef': 0.3,
        },
        {
            'buffer_size': 2,
            'num_steps': 2048,
            'ros_num_steps': 1024,
            'lr': 1e-4,
            'ros_lr': 1e-3,
            'update_epochs': 10,
            'ros_update_epochs': 8,
            'target_kl': 0.03,
            'ros_target_kl': 0.05,
            'ros_clip_coef': 0.3,
        },
        {
            'buffer_size': 2,
            'num_steps': 2048,
            'ros_num_steps': 1024,
            'lr': 1e-4,
            'ros_lr': 1e-3,
            'update_epochs': 10,
            'ros_update_epochs': 16,
            'target_kl': 0.03,
            'ros_target_kl': 0.05,
            'ros_clip_coef': 0.3,
        },
        {
            'buffer_size': 2,
            'num_steps': 2048,
            'ros_num_steps': 1024,
            'lr': 1e-4,
            'ros_lr': 1e-3,
            'update_epochs': 10,
            'ros_update_epochs': 32,
            'target_kl': 0.03,
            'ros_target_kl': 0.05,
            'ros_clip_coef': 0.3,
        },
    ]

    env_id = 'Hopper-v4'
    f = open(f"commands/ros_hopper_b2_50.txt", "w")
    for params in params_list:
        args = gen_args(
            device='cpu',
            length="short",
            arg_generator=ppo_ros,
            env_id=env_id,
            stats=0,
            anneal_lr=1,
            ros_lambda=0,
            **params,
        )
        write_to_file(f, args)

if __name__ == "__main__":

    write_args()

