
import sys
import os
import importlib
from logger import Logger


def main():
    stable_baselines3 = importlib.import_module("stable_baselines3")

    torch = importlib.import_module("torch")
    nn = torch.nn
    activation_fn = nn.Tanh

    torch.set_num_threads(2)

    env_class = importlib.import_module("panda.env").PandaEnv
    env_kwargs = {
        "scene": "/opt/robotics-rl/scenes/scene_panda.ttt",
        "headless": True,
        "joints": [0, 1, 2],
        "episode_length": 50,
        "log_dir": "/opt/results",
        "logger_class": Logger,
    }
    env = env_class(**env_kwargs)

    callback_class = importlib.import_module("panda.callback").CustomCallback
    callback_kwargs = {
        "n_steps": 100000,
        "save_path": "/opt/results/models"
    }
    callback = callback_class(**callback_kwargs)

    algorithm_class = stable_baselines3.TD3
    algorithm_kwargs = {
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "net_arch": [100, 100],
            "activation_fn": activation_fn,
        },
    }

    learn_kwargs = {
        "total_timesteps": 10000000
    }

    algorithm = algorithm_class(env=env, **algorithm_kwargs)
    algorithm.learn(**learn_kwargs, callback=callback)


if __name__ == "__main__":
    venv_path = os.environ["VENV"]
    sys.path.insert(0, venv_path)

    main()
