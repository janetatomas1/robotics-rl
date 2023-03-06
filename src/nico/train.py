
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import pathlib

from src.nico.env import NicoEnv
from src.callback import CustomCallback
from src.logger import CSVLogger


def train():
    joints = [
        "r_shoulder_y",
        "r_shoulder_z",
        "r_arm_x",
        "r_elbow_y",
        "r_arm_x",
        "r_wrist_z",
        "r_wrist_x",
        "r_indexfingers_x",
    ]

    root = pathlib.Path(__file__).parent.parent.parent

    scene = str(pathlib.Path(root, 'scenes', 'NICO-standing.ttt'))
    config = str(pathlib.Path(root, 'configs', 'nico_humanoid_vrep.json'))

    env = NicoEnv(
        joints=joints,
        scene=scene,
        config=config,
        log_file='/opt/results/values.csv',
        target_low = [0, -0.3, 0.5],
        target_high = [0.3, 0.3, 0.8],
        headless=False,
        reward_fn="boosted_reward",
        logger_class=CSVLogger,
        create_obstacles_fn='static'
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    callback = CustomCallback(n_steps=5000, save_path='/opt/results/models')

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=200000, callback=callback)

train()
