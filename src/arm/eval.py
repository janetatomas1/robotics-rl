import json

from stable_baselines3 import TD3
import pathlib
import torch
import glob

from .envs import PandaEnv
from src.logger import CSVLogger
import os
import json


def filename(m):
    return m.replace('models', 'eval').replace('zip', 'pickle')


def evaluate_model(env, model_file, positions, log_file):
    model = TD3.load(model_file)

    logger = env.get_logger()
    logger.open(log_file)

    for p in positions:
        env.reset_robot()
        env.set_starting_joint_values(p['starting_joint_positions'])
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
    episodes = 1000

    if not os.path.exists(path):
        os.mkdir(path)

    torch.set_num_threads(1)

    scene = pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'scenes', 'scene_panda.ttt')

    env_kwargs = {
        "scene": str(scene),
        "headless": False,
        "episode_length": 50,
        "reward_fn": "sparse_reward",
        "target_low": [0.8, -0.2, 1.0],
        "target_high": [1.0, 0.2, 1.4],
        "reset_actions": 5,
        "logger_class": CSVLogger,
        "dynamic_obstacles": False,
        "obstacles_state": [
            [0.5, 0.5, 0.6, 0.3, 0.4, 0.5],
            [0.5, 0.25, 0.75, 0.5, -0.25, 1.2],
        ],
    }

    env = PandaEnv(**env_kwargs)

    saved_models = glob.glob('/opt/results/models/*.zip')

    positions_file = open('/opt/positions/positions.json')
    positions = json.load(positions_file)
    positions_file.close()

    for m in saved_models:
        evaluate_model(env, m, positions, filename(m))

    env.close()
