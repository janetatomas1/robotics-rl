
import os
import git

from src.arm.learn import (
    train,
    evaluate
)

if __name__ == "__main__":
    venv_path = os.environ["VENV"]

    with open("/opt/results/git.txt", 'w') as f:
        repo = git.Repo(search_parent_directories=True)
        f.write("branch: {}\n".format(repo.active_branch.name))
        f.write("commit: {}\n".format(repo.head.object.hexsha))

    train()
    evaluate()
