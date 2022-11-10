
import sys
import os
import importlib


if __name__ == "__main__":
    venv_path = os.environ["VENV"]
    sys.path.insert(0, venv_path)

    main = importlib.import_module("panda.train").main

    main()
