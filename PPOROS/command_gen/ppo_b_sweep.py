import os

from PPOROS.scripts.utils import gen_args, ppo_buffer, write_to_file

if __name__ == "__main__":
    env_ids = [ 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'Ant-v4', 'Humanoid-v4']
    # env_ids = [ 'Walker2d-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'Ant-v4', 'Humanoid-v4']

    env_ids = [ 'Hopper-v4']

    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for history in [1]:
            f = open(f"commands/ppo_b{history}.txt", "w")
            for lr in [1e-3, 1e-4]:
                num_steps_list = [1024, 2048, 4096, 8192]
                # if env_id in ['Humanoid-v4']:
                #     num_steps_list = [4096, 8192]
                for num_steps in num_steps_list:
                    for target_kl in [0.03]:
                        epochs = 10
                        args = gen_args(
                            device='cpu',
                            length="short",
                            arg_generator=ppo_buffer,
                            env_id=env_id,
                            lr=lr,
                            epochs=epochs,
                            num_steps=num_steps,
                            buffer_size=history,
                            target_kl=target_kl,
                            df=2,
                            se=0,
                        )
                        write_to_file(f, args)

