import git
import torch
import pathlib
from stable_baselines3 import (
    TD3,
    SAC,
    PPO,
    DDPG,
)

from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch.nn as nn
import json
import os
import argparse

from roboticsrl.arm_env import PandaEnv

scene = pathlib.Path(pathlib.Path(__file__).parent.parent, 'scenes', 'scene_panda.ttt')
algorithms = {
    "td3": TD3,
    "ddpg": DDPG,
    "sac": SAC,
    "ppo": PPO,
}

env_kwargs = {
    "scene": str(scene),
    "headless": "HEADLESS" in os.environ and int(os.environ["HEADLESS"]) == 1,
    "episode_length": 50,
    "reward_fn": "boosted_sparse_reward",
    "target_low": [0.8, -0.2, 1.0],
    "target_high": [1.0, 0.2, 1.4],
    "reset_actions": 5,
    "dynamic_obstacles": False,
    "success_reward": 20,
    "max_speed": 0.2,
}


def train(args):
    torch.set_num_threads(args.threads)
    env = PandaEnv(**env_kwargs, log_file=f'{args.results_directory}/train_values.txt')

    algorithm_kwargs = {
        "env": env,
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "net_arch": [100, 100],
            "activation_fn": nn.Tanh,
        },
    }

    if args.algorithm != "ppo":
        n_actions = len(env.get_joints())
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        algorithm_kwargs["action_noise"] = action_noise

    model = algorithms[args.algorithm](**algorithm_kwargs)
    model.learn(total_timesteps=500000)
    model.save(f'{args.results_directory}/rl_model.zip')
    env.close()


def filename(m):
    return m.replace('zip', 'txt')


def evaluate_model_with_positions(env, model_file, positions, log_file):
    model = algorithms["algorithm"].load(model_file)
    logger = env.get_logger()
    logger.open(log_file)
    eval_runs = 10

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


def evaluate_model_without_positions(env, model, log_file):
    logger = env.get_logger()
    logger.open(log_file)
    episodes = 10000
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


def evaluate(args):
    env = PandaEnv(**env_kwargs)

    model_file = f'{args.results_directory}/rl_model.zip'
    model = algorithms[args.algorithm].load(model_file)
    log_file = f'{args.results_directory}/eval_values.txt'

    if args.positions_file == "none":
        evaluate_model_without_positions(env, model, log_file)
    else:
        positions_file = open(args.positions_file)
        positions = json.load(positions_file)
        positions_file.close()
        evaluate_model_with_positions(env, model, positions, log_file)
    env.close()


def save_git_info(directory):
    with open(f"{directory}/git.txt", 'w') as f:
        repo = git.Repo(search_parent_directories=True)
        f.write("branch: {}\n".format(repo.active_branch.name))
        f.write("commit: {}\n".format(repo.head.object.hexsha))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--results-directory',
        "-rd",
        default="/opt/results",
        help="Path to the directory where the results should be saved"
    )
    parser.add_argument(
        '--algorithm',
        '-a',
        choices=[
            'ddpg', 'td3', 'sac', 'ppo',
        ],
        help='The training algorihtm to be used',
        default='sac'
    )
    parser.add_argument(
        '--settings-file',
        '-sf',
        default='settings/no_obstacles.json',
        help='The path of the file specifying details of the experiment',
    )
    parser.add_argument(
        '--positions-file',
        '-pd',
        default='positions/positions.json',
        help='The path of the file containing the testing data. Should be used only for experiments without '
             'obstacles. To disable this option, set to "none"'
    )
    parser.add_argument(
        '--threads',
        default=1,
        help='Number of threads to be used for training'
    )
    parser.add_help = True

    args = parser.parse_args()
    if hasattr(args, 'help'):
        parser.print_help()
        parser.exit(0)

    save_git_info(args.results_directory)
    train(args)
    evaluate(args)
