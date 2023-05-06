
from stable_baselines3 import (
    TD3,
    SAC,
    PPO,
    DDPG,
)
import pathlib
import os
import argparse

from roboticsrl.arm_env import PandaEnv
import time

def demo(algorithm):
    scene = pathlib.Path(pathlib.Path(__file__).parent.parent, 'scenes', 'scene_panda.ttt')
    algorithms = {
        "td3": TD3,
        "ddpg": DDPG,
        "sac": SAC,
        "ppo": PPO,
    }

    model_file = pathlib.Path(pathlib.Path(__file__).parent.parent, 'models', f'{algorithm}_no_obstacles.zip')
    model = algorithms[algorithm].load(str(model_file))

    env_kwargs = {
        "scene": str(scene),
        "headless": "HEADLESS" in os.environ and int(os.environ["HEADLESS"]) == 1,
        "episode_length": 50,
        "reward_fn": "joint_sparse_reward",
        "target_low": [0.8, -0.2, 1.0],
        "target_high": [1.0, 0.2, 1.4],
        "reset_actions": 5,
        "dynamic_obstacles": False,
        "success_reward": 20,
        "max_speed": 0.2,
    }
    env = PandaEnv(**env_kwargs)
    episodes = 10

    for i in range(episodes):
        obs = env.reset()

        while not env.is_done():
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)
            time.sleep(0.3)

        print(f'Episode: {i}, joint path length: {env.path_cost()}, cartesian path length: {env.tip_path_cost()}, success: {env.is_close()}')

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        "-a",
        help="Algorithm to use",
        choices=["ddpg", "td3", "sac", "ppo"],
        default="sac"
    )

    args = parser.parse_args()

    demo(args.algorithm)