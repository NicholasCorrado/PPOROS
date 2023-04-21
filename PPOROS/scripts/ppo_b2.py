from PPOROS.hyperparams.ppo_buffer import PPOBUFFER_PARAMS
from PPOROS.scripts.utils import gen_args, ppo_buffer, write_to_file

if __name__ == "__main__":
    env_ids = [ 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'Ant-v4', 'Humanoid-v4']

    f = open(f"args/ppo_buffer.txt", "w")
    for env_id in env_ids:
        for history in [2]:
            for target_kl in [0.1]:
                lr = PPOBUFFER_PARAMS[env_id][history]['lr']
                num_steps = PPOBUFFER_PARAMS[env_id][history]['num_steps']

                epochs = 10
                args = gen_args(
                    device='cpu',
                    length="short",
                    arg_generator=ppo_buffer,
                    env_id=env_id,
                    lr=lr,
                    epochs=epochs,
                    num_steps=num_steps,
                    ros_history=history,
                    target_kl=target_kl,
                    stats=0,
                )
                write_to_file(f, args)

