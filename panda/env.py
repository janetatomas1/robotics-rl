
from gym import spaces, Env
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np


class PandaEnv(Env):
    def __init__(self, scene, threshold=0.3, joints=None, episode_length=100, headless=False):
        self.threshold = threshold
        self.headless = headless
        self.episode_length = episode_length
        self.steps = 0
        self.scene = scene
        self.pyrep = PyRep()
        self.pyrep.launch(scene_file=self.scene, headless=headless)
        self.pyrep.start()
        self.robot = Panda()
        self.robot.set_control_loop_enabled(False)
        self.initial_joint_positions = self.robot.get_joint_positions()
        self.robot.set_motor_locked_at_zero_velocity(True)
        self.joints = joints if joints is not None \
            else [i for i in range(len(self.robot.joints))]
        self.target = Shape('target')

        _, joint_intervals = self.robot.get_joint_intervals()
        low = [joint_intervals[j][0] for j in self.joints]
        high = [joint_intervals[j][1] for j in self.joints]

        self.observation_space = spaces.Box(
            low=np.array([-1., -1., 0.] + low),
            high=np.array([1., 1., 1.] + high),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=np.array([-2 for j in self.joints]),
            high=np.array([2 for j in self.joints])
        )
        self.timestep = self.pyrep.get_simulation_timestep()

    def reset(self):
        state = self.observation_space.sample()
        joint_positions = self.robot.get_joint_positions()

        for j, x in zip(self.joints, state[3:]):
            joint_positions[j] = x

        self.target.set_position(state[:3])
        self.robot.set_joint_positions(self.initial_joint_positions)
        # self.pyrep.step()
        return self._get_state()

    def render(self, mode="human"):
        pass

    def _get_state(self):
        return np.concatenate([self.target.get_position(),
                               [self.robot.get_joint_positions()[j] for j in self.joints]])

    def _is_close(self):
        return bool(np.linalg.norm(self.target.get_position() - self.robot.get_tip().get_position())
                    <= self.threshold)

    def _reward(self):
        return -np.linalg.norm(self.target.get_position() - self.robot.get_tip().get_position())

    def _is_done(self):
        return self._is_close() or self.steps >= self.episode_length

    def _info(self):
        return {}

    def step(self, action):
        self.steps += 1

        next_state = self._get_state()[3:] + self.timestep * action
        if self.observation_space.contains(next_state):
            return self._get_state(), -100, False, self._info()

        for j, v in zip(self.joints, action):
            self.robot.joints[j].set_joint_target_velocity(v)

        self.pyrep.step()

        done = self._is_done()

        if done:
            self.steps = 0

        return self._get_state(), self._reward(), done, self._info()
