
from gym import spaces, Env
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np
import torch


class PandaEnv(Env):
    def __init__(self, scene, threshold=0.3, joints=None, episode_length=100,
                 headless=False, reset_actions=10, log_dir=None, logger_class=None):
        self.robot = None
        self.target = None
        self.reset_actions = reset_actions
        self.threshold = threshold
        self.headless = headless
        self.log_dir = log_dir
        self.episode_length = episode_length
        self.steps = 0
        self.scene = scene
        self.pyrep = PyRep()

        self.pyrep.launch(scene_file=self.scene, headless=headless)
        self.restart_simulation()
        self.joints = joints if joints is not None \
            else [i for i in range(len(self.robot.joints))]

        _, joint_intervals = self.robot.get_joint_intervals()
        low = [joint_intervals[j][0] for j in self.joints]
        high = [joint_intervals[j][1] for j in self.joints]

        self.observation_space = spaces.Box(
            low=np.array([0.8, -0.2, 0.5] + low),
            high=np.array([1.0, 0.2, 1.4] + high),
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

    def restart_simulation(self):
        self.pyrep.stop()
        self.pyrep.start()

        self.robot = Panda()
        self.target = Shape('target')

        self.robot.set_control_loop_enabled(False)
        self.robot.set_motor_locked_at_zero_velocity(True)

    def move(self, action):
        for j, v in zip(self.joints, action):
            self.robot.joints[j].set_joint_target_velocity(v)

        self.pyrep.step()

    def reset(self):
        self.restart_simulation()
        self.steps = 0

        state = self.observation_space.sample()
        self.target.set_position(state[:3])

        for _ in range(self.reset_actions):
            action = self.action_space.sample()
            self.move(action)

        return self._get_state()

    def render(self, mode="human"):
        pass

    def _get_state(self):
        return np.concatenate([self.target.get_position(),
                               [self.robot.get_joint_positions()[j] for j in self.joints]])

    def _is_close(self):
        return bool(np.linalg.norm(self.target.get_position() - self.robot.get_tip().get_position())
                    <= self.threshold)

    def _reward(self, done=False, close=False):
        if not done:
            return -np.linalg.norm(self.target.get_position() - self.robot.get_tip().get_position())

        if done and not close:
            return -100

        if close:
            return 100

    def _is_done(self):
        return self._is_close() or self.steps >= self.episode_length

    def _info(self):
        return {}

    def step(self, action):
        self.steps += 1

        next_state = self._get_state()[3:] + self.timestep * action
        if self.observation_space.contains(next_state):
            return self._get_state(), -100, False, self._info()

        self.move(action)

        done = self._is_done()
        close = self._is_close()
        reward = self._reward()

        if self.logger is not None:
            self.logger.step(reward, done, close)

        return self._get_state(), reward, done, self._info()

    def close(self):
        self.logger.stop()
