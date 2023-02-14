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
from src.utils import distance


class ArmEnv(RobotEnv):
    INFO = {}

    def __init__(self,
                 reset_actions=3,
                 joints=None,
                 max_speed=1,
                 **kwargs):
        super().__init__(**kwargs)
        self._joints = joints if joints is not None else [i for i in range(len(self._robot.joints))]
        self._nreset_actions = reset_actions
        self._reset_actions = list()
        self._tip_path = list()

        _, joint_intervals = self._robot.get_joint_intervals()

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

    def clear_history(self):
        super().clear_history()
        self._tip_path.clear()

    def update_history(self):
        super().update_history()
        self._path.append(self.get_joint_values())
        self._tip_path.append(self.get_robot().get_tip().get_position())

    def move(self, action):
        for j, v in zip(self._joints, action):
            self._robot.joints[j].set_joint_target_velocity(v)

        self._pyrep.step()

    def reset(self):
        super().reset()
        self._reset_actions.clear()
        self._robot.set_control_loop_enabled(False)
        self._robot.set_motor_locked_at_zero_velocity(False)
        state = self.observation_space.sample()
        self._target.set_position(state[:3])
        self._robot.set_joint_positions(state[3:])
        self.get_pyrep_instance().step()

        for _ in range(self._nreset_actions):
            action = self.action_space.sample()
            self._reset_actions.append(action)
            self.move(action)

        self.move(np.zeros((len(self._joints),)))
        return self.get_state()

    def get_joint_values(self):
        return np.array([self._robot.get_joint_positions()[j] for j in self._joints])

    def distance(self):
        return np.linalg.norm(np.array(self._robot.get_tip().get_position()) - np.array(self._target.get_position()))

    def get_state(self):
        return np.concatenate([self._target.get_position(), self.get_joint_values()])

    def get_joints(self):
        return self._joints

    def is_close(self):
        return bool(self.distance() <= self._threshold)

    def reward_boost(self):
        return self.BOOSTED_REWARD - distance(self.get_path())

    def info(self):
        return {}

    def get_reset_actions(self):
        return self._reset_actions

    def get_tip_path(self):
        return self._tip_path


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
