
import sys
import os
import json
import importlib
from logger import Logger
from datetime import datetime


def main():
    stable_baselines3 = importlib.import_module("stable_baselines3")

    torch = importlib.import_module("torch")
    nn = torch.nn

    git = importlib.import_module("git")
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    settings = json.loads(os.environ["settings"])
    remote = settings.get("remote", True)

    settings["details"] = {
        "timestamp": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "sha": sha,
    }

    if remote:
        torch.set_num_threads(2)

    with open("/opt/results/settings.json", "w") as json_file:
        json.dump(settings, fp=json_file, indent=2)

    algorithm_settings = settings["algorithm"]
    algorithm_name = algorithm_settings["name"]
    algorithm_kwargs = algorithm_settings["kwargs"]
    algorithm_learn_kwargs = algorithm_settings["learn_kwargs"]
    algorithm_class = getattr(stable_baselines3, algorithm_name)

    policy_kwargs = algorithm_kwargs["policy_kwargs"]
    policy_kwargs["activation_fn"] = nn.Tanh
    algorithm_kwargs["policy_kwargs"] = policy_kwargs

    rl_env_settings = settings["rl_env"]
    rl_env_module_name = rl_env_settings["module"]
    rl_env_name = rl_env_settings["name"]
    rl_env_kwargs = rl_env_settings["kwargs"]
    rl_env_module = importlib.import_module(rl_env_module_name)
    rl_env_class = getattr(rl_env_module, rl_env_name)
    rl_env = rl_env_class(**rl_env_kwargs, logger_class=Logger)

    callback_settings = settings["callback"]
    callback_module_name = callback_settings["module"]
    callback_name = callback_settings["name"]
    callback_module = importlib.import_module(callback_module_name)
    callback_class = getattr(callback_module, callback_name)
    callback_kwargs = callback_settings["kwargs"]
    callback = callback_class(**callback_kwargs)

    algorithm = algorithm_class(policy="MlpPolicy", env=rl_env, **algorithm_kwargs)
    algorithm.learn(**algorithm_learn_kwargs, callback=callback)


if __name__ == "__main__":
    venv_path = os.environ["VENV"]
    sys.path.insert(0, venv_path)

    main()
