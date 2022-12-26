
from gym import Env
from pyrep import PyRep
from pyrep.objects.shape import Shape
import numpy as np


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
                 threshold=0.1,
                 episode_length=100,
                 headless=False,
                 log_dir=None,
                 logger_class=None,
                 reward_fn='sparse_reward'):
        self._threshold = threshold
        self._log_dir = log_dir
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

        if logger_class is not None:
            self.logger = logger_class("{}/values.txt".format(self._log_dir))
        else:
            self.logger = None
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

    def path_cost(self):
        cost = 0
        for i in range(1, len(self._path)):
            cost += np.linalg.norm(self._path[i] - self._path[i-1])

        return cost

    def boosted_sparse_reward(self):
        done = self.is_done()
        close = self.is_close()

        if close:
            return self.reward_boost()
        elif done:
            return self.FAILURE_REWARD

        return self.STEP_FAILURE_REWARD

    def reward_boost(self):
        return self.BOOSTED_REWARD - self.path_cost()

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

        self.update_history()

        if self.logger is not None:
            self.logger.step(reward, done, close, info)

        return self.get_state(), reward, done, info

    def close(self):
        if self.logger is not None:
            self.logger.stop()
        self._pyrep.stop()
        self._pyrep.shutdown()

    def get_pyrep_instance(self):
        return self._pyrep

    def get_history(self):
        return self._path, self._rewards
