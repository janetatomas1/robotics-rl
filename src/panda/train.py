
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch
import torch.nn as nn
import pathlib


from src.logger import Logger
from src.callback import CustomCallback
from .env import (
    PandaEnv,
    sparse_reward,
    punish_long_path_reward,
)


def train():
    joints = [0, 1, 2, 3, 4, 5, 6]
    n_actions = len(joints)
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    torch.set_num_threads(1)

    scene = pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'scenes', 'scene_panda.ttt')

    env_kwargs = {
        "scene": str(scene),
        "headless": True,
        "joints": joints,
        "episode_length": 50,
        "log_dir": "/opt/results",
        "logger_class": Logger,
        "reward_fn": punish_long_path_reward,
    }
    env = PandaEnv(**env_kwargs)

    callback_kwargs = {
        "n_steps": 10000,
        "save_path": "/opt/results/models"
    }
    callback = CustomCallback(**callback_kwargs)

    algorithm_kwargs = {
        "env": env,
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "net_arch": [100, 100],
            "activation_fn": nn.Tanh,
        },
        "action_noise": action_noise,
    }

    learn_kwargs = {
        "callback": callback,
        "total_timesteps": 10000000
    }

    algorithm = TD3(**algorithm_kwargs)
    algorithm.learn(**learn_kwargs)

    return algorithm


if __name__ == "__main__":
    train()
