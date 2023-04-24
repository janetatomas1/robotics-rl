
import torch
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch.nn as nn
import json

import os
from src.arm.envs import PandaEnv
from src.callback import CustomCallback


scene = pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'scenes', 'scene_panda.ttt')

algorithm_class = PPO
eval_runs = 10


def get_env(training):
    env_kwargs = {
        "scene": str(scene),
        "headless": "HEADLESS" in os.environ and int(os.environ["HEADLESS"]) == 1,
        "episode_length": 50,
        "log_file": f"/opt/results/values{'' if training else 1}.json",
        "reward_fn": "tip_boosted_sparse_reward",
        "target_low": [0.8, -0.2, 1.0],
        "target_high": [1.0, 0.2, 1.4],
        "reset_actions": 5,
        "dynamic_obstacles": False,
        "success_reward": 20,
        "max_speed": 0.2,
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

    for p in positions:

        cost = list()
        tip_cost = list()
        success = list()
        steps = list()
        collisions = list()
        quaternion_angle_costs = list()
        path_angle_costs = list()

        for i in range(eval_runs):
            env.empty_move()
            env.set_reset_joint_values(p['joints'])
            env.reset_joint_values()
            env.get_target().set_position(p['target_pos'])

            obs = env.get_state()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, done, _ = env.step(action)

            cost.append(env.path_cost())
            tip_cost.append(env.tip_path_cost())
            success.append(env.is_close())
            steps.append(env.get_steps())
            collisions.append(env.get_collision_count())
            quaternion_angle_costs.append(env.quaternion_angle_cost())
            path_angle_costs.append(env.joint_path_angle_cost())
            env.clear_history()

        env.save_history(history=dict(
            distance=cost,
            tip_distance=tip_cost,
            success=success,
            id_=p['id_'],
            steps=steps,
            collisions=collisions,
            quaternion_angle_costs=quaternion_angle_costs,
            path_angle_costs=path_angle_costs,
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
