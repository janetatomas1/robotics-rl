import pathlib

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EveryNTimesteps,
)


class CustomCallback(EveryNTimesteps):
    def __init__(self, n_steps, save_path):
        self.path = pathlib.Path(save_path)
        self.path.mkdir(parents=True, exist_ok=True)

        callback = CheckpointCallback(save_path=save_path, save_freq=1)
        super().__init__(n_steps, callback=callback)
