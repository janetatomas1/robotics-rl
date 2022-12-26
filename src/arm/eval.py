
from stable_baselines3 import TD3
import numpy as np
import torch
import torch.nn as nn
import pathlib

from .envs import PandaEnv


def rl_path(env, model):
    obs = env.get_state()
    done = False

    while not done:
        action = model.predict(obs)
        obs, _, done, _ = env.step(action)


def algo_path(env, algorithm):
    env.clear_history()
    target = env.get_target()
    robot = env.get_robot()

    path = robot.get_path(
        position=target.get_position(),
        quaternion=[0, 0, 1, 0],
        distance_threshold=0.1,
        trials=1000,
        max_configs=1000,
    )

    done = False
    env.update_history()



def evaluate():
    torch.set_num_threads(1)

    scene = pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'scenes', 'scene_panda.ttt')

    env_kwargs = {
        "scene": str(scene),
        "headless": False,
        "episode_length": 50,
        "log_dir": "/opt/results",
        "reward_fn": "sparse_reward",
        "target_low": [0.8, -0.2, 1.0],
        "target_high": [1.0, 0.2, 1.4],
        "reset_actions": 10,
        "with_quaternion": False,
    }

    env = PandaEnv(**env_kwargs)
    model = TD3.load('/home/janetatomas11/Desktop/results/yq41z869rgne05bfmatx/models/rl_model_560000_steps.zip')

    for i in range(10):
        obs = env.reset()
        reset_actions = env.get_reset_actions()

        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

