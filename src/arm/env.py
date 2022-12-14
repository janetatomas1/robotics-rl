
from gym import spaces, Env
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np


def sparse_reward(**kwargs):
    done = kwargs["done"]
    close = kwargs["close"]

    if close:
        return 1
    elif not done:
        return -1

    return -2


def path_cost(path):
    if len(path) <= 1:
        return 0

    cost = 0
    for i in range(1, len(path)):
        cost += np.linalg.norm(path[i-1][0] - path[i][0])

    return cost


def punish_long_path_reward(**kwargs):
    done = kwargs["done"]
    close = kwargs["close"]

    if close:
        path = kwargs["path"]
        return 20 - path_cost(path)
    elif not done:
        return -1

    return -2


def path_cost2(path, target_position, threshold):
    if len(path) <= 1:
        return 0

    cost = 0
    for i in range(1, len(path)):
        if np.linalg.norm(path[i][1] - target_position) <= threshold:
            cost += np.linalg.norm(path[i - 1][0] - path[i][0]) * 3
        else:
            cost += np.linalg.norm(path[i - 1][0] - path[i][0])

    return cost


def punish_long_path_reward2(**kwargs):
    done = kwargs["done"]
    close = kwargs["close"]
    target = kwargs["target"]
    threshold = kwargs["short_distance_threshold"]

    if close:
        path = kwargs["path"]
        return 20 - path_cost2(path, target.get_position(), threshold)
    elif not done:
        return -1

    return -2


class PandaEnv(Env):
    INFO = {}

    def __init__(self,
                 scene,
                 threshold=0.1,
                 short_distance_threshold=0.5,
                 joints=None,
                 episode_length=100,
                 target_low=None,
                 target_high=None,
                 headless=False,
                 reset_actions=10,
                 log_dir=None,
                 logger_class=None,
                 reward_fn=sparse_reward,
                 training=True):
        self.robot = None
        self.target = None
        self.reset_actions = reset_actions
        self.threshold = threshold
        self.short_distance_threshold = short_distance_threshold
        self.headless = headless
        self.log_dir = log_dir
        self.episode_length = episode_length
        self.scene = scene
        self.reward = reward_fn
        self.training = training
        self.steps = 0
        self.pyrep = PyRep()
        self.pyrep.launch(scene_file=self.scene, headless=headless)
        self.robot = Panda()
        self.target = Shape('target')
        self.restart_simulation()
        self.joints = joints if joints is not None \
            else [i for i in range(len(self.robot.joints))]

        _, joint_intervals = self.robot.get_joint_intervals()
        self.low = np.array([joint_intervals[j][0] for j in self.joints])
        self.high = np.array([joint_intervals[j][1] for j in self.joints])

        self.observation_space = spaces.Box(
            low=np.concatenate([[0.8, -0.2, 0.5] if target_low is None else target_low, self.low]),
            high=np.concatenate([[1.0, 0.2, 1.4] if target_high is None else target_high, self.high]),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=np.array([-2 for j in self.joints]),
            high=np.array([2 for j in self.joints])
        )
        self.timestep = self.pyrep.get_simulation_timestep()

        if logger_class is not None:
            self.logger = logger_class("{}/values.txt".format(self.log_dir))
        else:
            self.logger = None
        self.path = list()

    def restart_simulation(self):
        self.pyrep.stop()
        self.pyrep.start()

        self.robot.set_control_loop_enabled(False)
        self.robot.set_motor_locked_at_zero_velocity(False)

    def move(self, action):
        for j, v in zip(self.joints, action):
            self.robot.joints[j].set_joint_target_velocity(v)

        self.pyrep.step()

    def reset(self):
        self.restart_simulation()
        self.steps = 0
        self.move(np.zeros((len(self.joints),)))
        self.path.clear()

        state = self.observation_space.sample()
        self.target.set_position(state[:3])

        for _ in range(self.reset_actions):
            action = self.action_space.sample()
            self.move(action)

        return self.get_state()

    def render(self, mode="human"):
        pass

    def get_joint_values(self):
        return np.array([self.robot.get_joint_positions()[j] for j in self.joints])

    def get_state(self):
        return np.concatenate([self.target.get_position(), self.get_joint_values()])

    def is_close(self):
        return bool(np.linalg.norm(self.target.get_position() - self.robot.get_tip().get_position())
                    <= self.threshold)

    def is_done(self):
        return self.is_close() or self.steps >= self.episode_length

    def step(self, action):
        if self.training:
            self.steps += 1

        self.path.append((self.get_joint_values(), self.robot.get_tip().get_position()))
        self.move(action)
        done = self.is_done()
        close = self.is_close()

        if done:
            self.path.append((self.get_joint_values(), self.robot.get_tip().get_position()))

        reward_kwargs = {
            "env": self,
            "done": done,
            "close": close,
            "path": self.path,
            "target": self.target,
            "short_distance_threshold": self.short_distance_threshold,
        }
        reward = self.reward(**reward_kwargs)

        if self.logger is not None:
            self.logger.step(reward, done, close)

        return self.get_state(), reward, done, self.INFO

    def close(self):
        if self.logger is not None:
            self.logger.stop()
        self.pyrep.stop()
        self.pyrep.shutdown()
