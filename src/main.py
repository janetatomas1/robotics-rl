
import sys
import os
import json
import importlib


def main():
    stable_baselines3 = importlib.import_module("stable_baselines3")
    algorithm_settings = json.loads(os.environ["algorithm_settings"])
    algorithm_name = algorithm_settings["name"]
    algorithm_kwargs = algorithm_settings["kwargs"]
    algorithm_train_kwargs = algorithm_settings["train_kwargs"]
    algorithm_class = getattr(stable_baselines3, algorithm_name)

    policy_settings = json.loads(os.environ["policy_settings"])
    policy_module_name = policy_settings["module"]
    policy_module = importlib.import_module(policy_module_name)
    policy_name = policy_settings["name"]
    policy_class = getattr(policy_module, policy_name)

    rl_env_settings = json.loads(os.environ["rl_env_settings"])
    rl_env_module_name = rl_env_settings["module"]
    rl_env_name = rl_env_settings["name"]
    rl_env_kwargs = rl_env_settings["kwargs"]
    rl_env_module = importlib.import_module(rl_env_module_name)
    rl_env_class = getattr(rl_env_module, rl_env_name)
    rl_env = rl_env_class(**rl_env_kwargs)

    algorithm = algorithm_class(policy="MlpPolicy", env=rl_env, **algorithm_kwargs)
    algorithm.learn(**algorithm_train_kwargs)


if __name__ == "__main__":
    venv_path = os.environ["VENV"]
    sys.path.insert(0, venv_path)

    main()
