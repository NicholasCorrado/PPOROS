import os

from PPOROS.hyperparams.ppo_buffer import PPOBUFFER_PARAMS
from PPOROS.scripts.utils import gen_args, ppo_buffer, write_to_file

if __name__ == "__main__":
    env_ids = [ 'Walker2d-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'Ant-v4', 'Humanoid-v4']
    # env_ids = [ 'Walker2d-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'Ant-v4', 'Humanoid-v4']

    os.makedirs('commands', exist_ok=True)
    for env_id in env_ids:
        for b in [2, 4]:
            PARAMS = PPOBUFFER_PARAMS[env_id][b]

            f = open(f"commands/ppo_b{b}.txt", "w")
            epochs = 10
            args = gen_args(
                device='cpu',
                length="short",
                arg_generator=ppo_buffer,
                env_id=env_id,
                epochs=epochs,
                buffer_size=b,
                se=1,
                **PARAMS,
            )
            write_to_file(f, args)

