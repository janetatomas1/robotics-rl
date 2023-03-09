from gym import Env
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
import numpy as np

from src.utils import distance
from src.logger import BinaryLogger


class RobotEnv(Env):
    BOOSTED_REWARD = 20
    SUCCESS_REWARD = 1
    COLLISION_REWARD = -3
    FAILURE_REWARD = -5
    STEP_FAILURE_REWARD = -1

    def __init__(self,
                 scene,
                 target_low,
                 target_high,
                 robot_class=None,
                 pr=None,
                 log_file=None,
                 threshold=0.1,
                 episode_length=50,
                 headless=False,
                 logger_class=BinaryLogger,
                 reward_fn='sparse_reward',
                 dynamic_obstacles=False,
                 obstacles_low=None,
                 obstacles_high=None,
                 obstacles_state=None,
                 obstacles_type=PrimitiveShape.CUBOID,
                 ):
        self._threshold = threshold
        self._log_file = log_file
        self._episode_length = episode_length
        self._scene = scene
        self._steps = 0
        self._target_low = target_low
        self._target_high = target_high
        self._dynamic_obstacles = dynamic_obstacles
        self._obstacles_low = np.array([] if obstacles_low is None else np.concatenate(obstacles_low))
        self._obstacles_high = np.array([] if obstacles_high is None else np.concatenate(obstacles_high))
        self._obstacles_state = np.array([] if obstacles_state is None else np.concatenate(obstacles_state))
        self._obstacle_type = obstacles_type
        self._obstacle_color = [0, 0, 1]
        self._obstacles_number = 0

        if self._dynamic_obstacles:
            self._obstacles_number = len(obstacles_low)
        elif obstacles_state is not None:
            self._obstacles_number = len(obstacles_state)

        self._obstacles = list()

        self._rewards = list()
        self._reward_fn = getattr(self, reward_fn)

        if pr is None:
            self._pyrep = PyRep()
            self._pyrep.launch(scene_file=self._scene, headless=headless)
            self._pyrep.start()
        else:
            self._pyrep = pr

        if robot_class is not None:
            self._robot = robot_class()
            self._initial_robot_state = self._robot.get_configuration_tree()

        self._target = Shape('target')
        self.create_obstacles()
        self._collision_count = 0

        for o in self._obstacles:
            o.set_collidable(True)

        self._logger = logger_class()

        if self._log_file is not None:
            self._logger.open(self._log_file)

        self._path = list()

    def reset_robot(self):
        self._pyrep.set_configuration_tree(self._initial_robot_state)

    def move(self, action):
        pass

    def sparse_reward(self):
        close = self.is_close()
        done = self.is_done()
        punishment = self.COLLISION_REWARD if self.check_collision() else 0

        if close:
            return self.SUCCESS_REWARD + punishment
        elif done:
            return self.FAILURE_REWARD + punishment

        return self.STEP_FAILURE_REWARD + punishment

    def boosted_reward(self):
        return -self.distance()

    def boosted_sparse_reward(self):
        done = self.is_done()
        close = self.is_close()
        punishment = self.COLLISION_REWARD if self.check_collision() else 0

        if close:
            return self.reward_boost() + punishment
        elif done:
            return self.FAILURE_REWARD + punishment

        return self.STEP_FAILURE_REWARD + punishment

    def reward_boost(self):
        return self.BOOSTED_REWARD - distance(self._path)

    def reset(self):
        self.reset_robot()
        self.reset_target()
        self.clear_history()

        if self._dynamic_obstacles:
            self.clear_obstacles()
            self.create_obstacles()

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
        self._collision_count = 0

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
                self.save_history(
                    collision_count=self._collision_count,
                    close=close,
                    rewards=self._rewards,
                    info={'info': info}
                )
            else:
                self.save_history(
                    close=close,
                    rewards=self._rewards,
                    collision_count=self._collision_count
                )

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

    def get_obstacles(self):
        return self._obstacles

    def check_collision(self):
        return False

    def clear_obstacles(self):
        for o in self._obstacles:
            o.remove()

    def create_obstacles(self):
        if self._dynamic_obstacles:
            self._obstacles_state = np.random.uniform(self._obstacles_low, self._obstacles_high)

        self._obstacles = [
            Shape.create(
                type=self._obstacle_type,
                color=self._obstacle_color,
                size=self._obstacles_state[(6 * i):(6 * i + 3)].tolist(),
                position=self._obstacles_state[(6 * i + 3):(6 * i + 6)].tolist(),
            ) for i in range(self._obstacles_number)
        ]

    def reset_target(self):
        done = False

        while not done:
            position = np.random.uniform(self._target_low, self._target_high)
            self.get_target().set_position(position=position)
            self.get_pyrep_instance().step()
            done = not any([self._target.check_collision(o) for o in self._obstacles])
