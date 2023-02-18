
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch
import torch.nn as nn
import pathlib

from src.logger import CSVLogger
from src.callback import CustomCallback
from .envs import (
    PandaEnv,
    JacoEnv,
    MicoEnv,
    UR3Env,
    UR5Env,
    UR10Env,
    LBRIwaa7R800Env,
    LBRIwaa14R820Env,
)


def train():
    torch.set_num_threads(1)

    scene = pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'scenes', 'scene_panda.ttt')

    env_kwargs = {
        "scene": str(scene),
        "headless": False,
        "episode_length": 50,
        "log_file": "/opt/results/values.csv",
        "reward_fn": "boosted_sparse_reward",
        "target_low": [0.8, -0.2, 1.0],
        "target_high": [1.0, 0.2, 1.4],
        "reset_actions": 5,
        "logger_class": CSVLogger,
    }

    env = PandaEnv(**env_kwargs)
    n_actions = len(env.get_joints())
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

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
        "total_timesteps": 500000
    }

    algorithm = TD3(**algorithm_kwargs)
    algorithm.learn(**learn_kwargs)

    env.close()
