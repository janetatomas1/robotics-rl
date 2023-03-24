
import torch
import pathlib
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch.nn as nn
import os
import json
import glob

from src.logger import CSVLogger
from src.arm.envs import PandaEnv
from src.callback import CustomCallback


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
    "dynamic_obstacles": False,
}

algorithm_class = TD3


def get_env():
    env = PandaEnv(**env_kwargs)
    env.set_control_loop(False)
    return env


def train():
    torch.set_num_threads(1)
    env = get_env()

    callback_kwargs = {
        "n_steps": 10000,
        "save_path": "/opt/results/models"
    }

    callback = CustomCallback(**callback_kwargs)

    n_actions = len(env.get_joints())
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

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
        "total_timesteps": 500000,
    }

    model = algorithm_class(**algorithm_kwargs)
    model.learn(**learn_kwargs)

    env.close()


def filename(m):
    return m.replace('models', 'eval').replace('zip', 'pickle')


def evaluate_model(env, model_file, positions, log_file):
    model = algorithm_class.load(model_file)

    logger = env.get_logger()
    logger.open(log_file)

    for p in positions:
        env.empty_move()
        env.set_reset_joint_values(p['joints'])
        env.reset_joint_values()
        env.get_target().set_position(p['target_pos'])

        obs = env.get_state()
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

        env.save_history(
            distance=env.path_cost(),
            tip_distance=env.tip_path_cost(),
            success=env.is_close(),
            id_=p['id_'],
        )

    logger.close()


def evaluate():
    path = '/opt/results/eval'

    if not os.path.exists(path):
        os.mkdir(path)

    torch.set_num_threads(1)

    env = get_env()

    saved_models = glob.glob('/opt/results/models/*.zip')

    positions_file = open('/opt/positions/positions.json')
    positions = json.load(positions_file)
    positions_file.close()

    for m in saved_models:
        evaluate_model(env, m, positions, filename(m))

    env.close()
