
from gym import Env
from pyrep import PyRep
from pyrep.objects.shape import Shape

from src.utils import distance
from src.logger import BinaryLogger


class RobotEnv(Env):
    BOOSTED_REWARD = 20
    SUCCESS_REWARD = 1
    FAILURE_REWARD = -2
    STEP_FAILURE_REWARD = -1

    def __init__(self,
                 scene,
                 robot_class,
                 target_low,
                 target_high,
                 log_file=None,
                 threshold=0.1,
                 episode_length=50,
                 headless=False,
                 logger_class=BinaryLogger,
                 reward_fn='sparse_reward'):
        self._threshold = threshold
        self._log_file = log_file
        self._episode_length = episode_length
        self._scene = scene
        self._steps = 0
        self._target_low = target_low
        self._target_high = target_high

        self._rewards = list()
        self._reward_fn = getattr(self, reward_fn)

        self._pyrep = PyRep()
        self._pyrep.launch(scene_file=self._scene, headless=headless)
        self._robot = robot_class()
        self._target = Shape('target')
        self.restart_simulation()

        self._logger = logger_class()

        if self._log_file is not None:
            self._logger.open(self._log_file)

        self._path = list()

    def restart_simulation(self):
        self._pyrep.stop()
        self._pyrep.start()

    def move(self, action):
        pass

    def sparse_reward(self):
        close = self.is_close()
        done = self.is_done()

        if close:
            return self.SUCCESS_REWARD
        elif done:
            return self.FAILURE_REWARD

        return self.STEP_FAILURE_REWARD

    def boosted_sparse_reward(self):
        done = self.is_done()
        close = self.is_close()

        if close:
            return self.reward_boost()
        elif done:
            return self.FAILURE_REWARD

        return self.STEP_FAILURE_REWARD

    def reward_boost(self):
        return self.BOOSTED_REWARD - distance(self._path)

    def reset_specific(self):
        pass

    def reset(self):
        self.restart_simulation()
        self.clear_history()
        return self.get_state()

    def render(self, mode="human"):
        pass

    def distance(self):
        return 0

    def get_state(self):
        pass

    def get_robot(self):
        return self._robot

    def get_target(self):
        return self._target

    def is_close(self):
        return bool(self.distance() <= self._threshold)

    def is_done(self):
        return self.is_close() or self._steps >= self._episode_length

    def info(self):
        return {}

    def clear_history(self):
        self._path.clear()
        self._rewards.clear()
        self._steps = 0

    def update_history(self):
        self._rewards.append(self._reward_fn())


    def step(self, action):
        self._steps += 1
        self.update_history()

        self.move(action)
        done = self.is_done()
        close = self.is_close()
        reward = self._reward_fn()
        info = self.info()

        if done:
            self.update_history()

        if done and self._log_file is not None:
            if len(info) > 0:
                self.save_history(close=close, rewards=self._rewards, info={'info': info})
            else:
                self.save_history(close=close, rewards=self._rewards)

        return self.get_state(), reward, done, info

    def close(self):
        self._logger.close()
        self._pyrep.stop()
        self._pyrep.shutdown()

    def save_history(self, **kwargs):
        self._logger.save_history(**kwargs)

    def get_pyrep_instance(self):
        return self._pyrep

    def get_path(self):
        return self._path

    def get_rewards(self):
        return self._rewards

    def get_logger(self):
        return self._logger
