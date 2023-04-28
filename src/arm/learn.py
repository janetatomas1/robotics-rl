
import torch
import pathlib
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch.nn as nn
import json

import os
from src.arm.envs import PandaEnv
from src.callback import CustomCallback


scene = pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'scenes', 'scene_panda.ttt')

algorithm_class = SAC
eval_runs = 10


def get_env(training):
    env_kwargs = {
        "scene": str(scene),
        "headless": "HEADLESS" in os.environ and int(os.environ["HEADLESS"]) == 1,
        "episode_length": 50,
        "log_file": f"/opt/results/values{'' if training else 1}.json",
        "reward_fn": "boosted_sparse_reward",
        "target_low": [0.92, 0.1, 0.7],
        "target_high": [1.2, 0.3, 1.2],
        "reset_actions": 5,
        "dynamic_obstacles": False,
        "success_reward": 20,
        "max_speed": 0.2,
        "obstacles_state": [
            [0.01, 0.9, 0.4, 0.9, 0, 0.9],
        ],
    }
    env = PandaEnv(**env_kwargs, save_history=training)
    env.set_control_loop(False)
    return env


def train():
    torch.set_num_threads(1)
    env = get_env(True)

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
    return m.replace('models', 'eval').replace('zip', 'json')


def evaluate_model(env, model_file, positions, log_file):
    model = algorithm_class.load(model_file)

    logger = env.get_logger()
    logger.open(log_file)
    episodes = 1000
    for i in range(episodes):
        env.clear_history()
        obs = env.reset()

        while not env.is_done():
            action, _ = model.predict(obs)
            env.step(action)


        env.save_history(history=dict(
            rewards=env.get_rewards(),
            cost=env.path_cost(),
            tip_cost=env.tip_path_cost(),
            quaternion_cost=env.quaternion_angle_cost(),
            collisions=env.get_collision_count(),
            steps=env.get_steps(),
        ))

    logger.close()


def evaluate():
    path = '/opt/results/eval'

    if not os.path.exists(path):
        os.mkdir(path)

    torch.set_num_threads(1)

    env = get_env(False)

    saved_model = str(pathlib.Path('/opt/results/models/rl_model_500000_steps.zip'))

    positions_file = open('/opt/positions/positions.json')
    positions = json.load(positions_file)
    positions_file.close()

    evaluate_model(env, saved_model, positions, filename(saved_model))
    env.close()
