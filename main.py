import gym
import numpy as np

from stable_baselines3 import DDPG, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from os import path

from envs.panda import PandaEnv

env = PandaEnv(path.join('scenes', 'scene_reinforcement_learning_env.ttt'))

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
env = model.get_env()


