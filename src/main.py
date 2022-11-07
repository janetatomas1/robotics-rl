
import sys
import os
import importlib
from logger import Logger

joints = [0, 1, 2]


def main():
    stable_baselines3 = importlib.import_module("stable_baselines3")
    noise_class = stable_baselines3.noise.NormalActionNoise

    np = importlib.import_module("numpy")

    n_actions = len(joints)
    action_noise = noise_class(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    torch = importlib.import_module("torch")
    nn = torch.nn
    activation_fn = nn.Tanh

    torch.set_num_threads(2)

    env_module = importlib.import_module("panda.env")
    env_reward_fn = env_module.shape_reward
    env_class = env_module.PandaEnv
    env_kwargs = {
        "scene": "/opt/robotics-rl/scenes/scene_panda.ttt",
        "headless": True,
        "joints": joints,
        "episode_length": 50,
        "log_dir": "/opt/results",
        "logger_class": Logger,
        "reward_fn": env_reward_fn,
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
        "action_noise": action_noise,
    }

    learn_kwargs = {
        "total_timesteps": 10000000
    }

    algorithm = algorithm_class(env=env, action_noise=action_noise, **algorithm_kwargs)
    algorithm.learn(**learn_kwargs, callback=callback)


if __name__ == "__main__":
    venv_path = os.environ["VENV"]
    sys.path.insert(0, venv_path)

    main()
