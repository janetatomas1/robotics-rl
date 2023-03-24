
import os
import git
from multiprocessing import Process

from src.arm.learn import (
    train,
    evaluate,
)

if __name__ == "__main__":
    venv_path = os.environ["VENV"]

    with open("/opt/results/git.txt", 'w') as f:
        repo = git.Repo(search_parent_directories=True)
        f.write("branch: {}\n".format(repo.active_branch.name))
        f.write("commit: {}\n".format(repo.head.object.hexsha))

    train_proc = Process(target=train)
    eval_proc = Process(target=evaluate)

    train_proc.start()
    train_proc.join()

    eval_proc.start()
    eval_proc.join()
