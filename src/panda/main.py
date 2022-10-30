import pathlib
from stable_baselines3.td3 import TD3, policies

from env import PandaEnv

scene = pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'scenes', 'scene_panda.ttt')

env = PandaEnv(scene=str(scene))

model = TD3(env=env, policy="MlpPolicy")
model.learn(total_timesteps=100000)


