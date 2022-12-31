
from stable_baselines3 import TD3
import pathlib
import torch
import glob

from .envs import PandaEnv
from src.logger import BinaryLogger
import os


def filename(m):
    return m.replace('models', 'eval').replace('zip', 'pickle')


def evaluate_model(env, model_file, episodes, log_file):
    model = TD3.load(model_file)

    logger = env.get_logger()
    logger.open(log_file)

    for _ in range(episodes):
        obs = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

        env.save_history(
            path=env.get_path(),
            tip_path=env.get_tip_path(),
            target_position=env.get_target().get_position(),
            success=env.is_close(),
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
        "headless": True,
        "episode_length": 50,
        "reward_fn": "sparse_reward",
        "target_low": [0.8, -0.2, 1.0],
        "target_high": [1.0, 0.2, 1.4],
        "reset_actions": 5,
        "with_quaternion": False,
        "logger_class": BinaryLogger,
    }

    env = PandaEnv(**env_kwargs)

    saved_models = glob.glob('/opt/results/models/*.zip')

    for m in saved_models:
        evaluate_model(env, m, episodes, filename(m))

    env.close()

