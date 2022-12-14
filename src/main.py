
import sys
import os
import importlib

if __name__ == "__main__":
    venv_path = os.environ["VENV"]
    sys.path.insert(0, venv_path)

    git = importlib.import_module("git")

    with open("/opt/results/git.txt", 'w') as f:
        repo = git.Repo(search_parent_directories=True)
        f.write("branch: {}\n".format(repo.active_branch.name))
        f.write("commit: {}\n".format(repo.head.object.hexsha))

    train = importlib.import_module("src.arm").train
    logger_class = importlib.import_module("src.logger").Logger
    callback_class = importlib.import_module("src.callback").CustomCallback

    train()

