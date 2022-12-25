import numpy as np
from gym import spaces

from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.jaco import Jaco
from pyrep.robots.arms.mico import Mico
from pyrep.robots.arms.ur3 import UR3
from pyrep.robots.arms.ur5 import UR5
from pyrep.robots.arms.ur10 import UR10
from pyrep.robots.arms.lbr_iiwa_7_r800 import LBRIwaa7R800
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820

from src.robot_env import RobotEnv


class ArmEnv(RobotEnv):
    INFO = {}

    def __init__(self,
                 reset_actions=3,
                 joints=None,
                 max_speed=1,
                 quaternion_threshold=0.05,
                 with_quaternion=True,
                 **kwargs):
        super().__init__(**kwargs)
        self._joints = joints if joints is not None else [i for i in range(len(self._robot.joints))]
        self._reset_actions = reset_actions
        self._quaternion_threshold = quaternion_threshold
        self._with_quaternion = with_quaternion
        self._quaternion = np.zeros((4,)) if self._with_quaternion else None


        _, joint_intervals = self._robot.get_joint_intervals()

        if self._with_quaternion:
            self._low = np.concatenate([
                self._target_low,
                [joint_intervals[j][0] for j in self._joints],
                [-1, -1, -1, -1] if self._with_quaternion else [],
            ])
            self._high = np.concatenate([
                self._target_high,
                [joint_intervals[j][1] for j in self._joints],
                [1, 1, 1, 1] if self._with_quaternion else [],
            ])
        else:
            self._low = np.concatenate([
                self._target_low,
                [joint_intervals[j][0] for j in self._joints],
            ])
            self._high = np.concatenate([
                self._target_high,
                [joint_intervals[j][1] for j in self._joints],
            ])

        self.observation_space = spaces.Box(
            low=self._low,
            high=self._high,
            dtype=np.float64,
        )

        self.action_space = spaces.Box(
            low=np.array([-max_speed for _ in self._joints]),
            high=np.array([max_speed for _ in self._joints])
        )

    def update_path(self):
        self._path.append(self.get_joint_values())

    def move(self, action):
        for j, v in zip(self._joints, action):
            self._robot.joints[j].set_joint_target_velocity(v)

        self._pyrep.step()

    def reset_specific(self):
        self._robot.set_control_loop_enabled(False)
        self._robot.set_motor_locked_at_zero_velocity(False)
        state = self.observation_space.sample()
        self._target.set_position(state[:3])

        if self._with_quaternion:
            self._quaternion = state[-4:]

        for _ in range(self._reset_actions):
            action = self.action_space.sample()
            self.move(action)

        self.move(np.zeros((len(self._joints),)))

    def get_joint_values(self):
        return np.array([self._robot.get_joint_positions()[j] for j in self._joints])

    def distance(self):
        return np.linalg.norm(np.array(self._robot.get_tip().get_position()) - np.array(self._target.get_position()))

    def get_state(self):
        if self._with_quaternion:
            return np.concatenate([self._target.get_position(), self.get_joint_values(), self._quaternion])

        return np.concatenate([self._target.get_position(), self.get_joint_values()])

    def get_joints(self):
        return self._joints

    def get_desired_quaternion(self):
        return self._quaternion

    def get_quaternion(self):
        return self.get_target().get_quaternion(self.get_robot().get_tip())

    def quaternion_distance(self):
        return np.linalg.norm(self._quaternion - self.get_quaternion())

    def is_close(self):
        return bool(self.distance() <= self._threshold)

    def reward_boost(self):
        if self._with_quaternion:
            return self.BOOSTED_REWARD - self.path_cost() - self.quaternion_distance() * 3

        return self.BOOSTED_REWARD - self.path_cost()

    def info(self):
        if self._with_quaternion and self.is_done():
            return {"quaternion_distance": self.quaternion_distance()}

        return {}


class PandaEnv(ArmEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_class=Panda, **kwargs)


class JacoEnv(ArmEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_class=Jaco, **kwargs)


class MicoEnv(ArmEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_class=Mico, **kwargs)


class UR3Env(ArmEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_class=UR3, **kwargs)


class UR5Env(ArmEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_class=UR5, **kwargs)


class UR10Env(ArmEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_class=UR10, **kwargs)


class LBRIwaa7R800Env(ArmEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_class=LBRIwaa7R800, **kwargs)


class LBRIwaa14R820Env(ArmEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_class=LBRIwaa14R820, **kwargs)
