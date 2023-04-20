from PPOROS.scripts.utils import gen_args, ppo_buffer, write_to_file

if __name__ == "__main__":
    env_ids = [ 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'Ant-v4', 'Humanoid-v4']

    f = open(f"args/ppo_buffer.txt", "w")
    for env_id in env_ids:
        for lr in [1e-3, 1e-4]:
            for history in [4]:
                for num_steps in [1024, 2048, 4096, 8192]:
                    for target_kl in [0.1]:
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

