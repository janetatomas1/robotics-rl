import numpy as np
from gym import spaces
import math

from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.jaco import Jaco
from pyrep.robots.arms.mico import Mico
from pyrep.robots.arms.ur3 import UR3
from pyrep.robots.arms.ur5 import UR5
from pyrep.robots.arms.ur10 import UR10
from pyrep.robots.arms.lbr_iiwa_7_r800 import LBRIwaa7R800
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820
from pyrep.const import ConfigurationPathAlgorithms
from pyrep.errors import ConfigurationPathError

from src.robot_env import RobotEnv
from src.utils import distance


class ArmEnv(RobotEnv):
    MAX_CONFIGS_DEFAULT = 100
    TRIALS_DEFAULT = 1
    MAX_TIME_MS_DEFAULT = 1
    ALGORITHM_DEFAULT = ConfigurationPathAlgorithms.PRM
    TRIALS_DEFAULT = 1
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

        self._starting_joint_positions = self.get_robot().get_joint_positions()

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
        state = self.observation_space.sample()
        self._target.set_position(state[:3])
        self.get_pyrep_instance().step()

        for _ in range(self._nreset_actions):
            self._reset_actions.append(self.action_space.sample())

        self.play_reset_actions()
        self.empty_move()

        return self.get_state()

    def restart_simulation(self):
        super().restart_simulation()
        self.get_robot().set_joint_positions(self._starting_joint_positions)
        self.get_robot().set_control_loop_enabled(False)
        self.get_robot().set_motor_locked_at_zero_velocity(False)
        self.get_robot().set_joint_target_positions(self._starting_joint_positions)
        self.empty_move()
        self.get_pyrep_instance().step()

    def play_reset_actions(self):
        for v in self._reset_actions:
            self.move(v)

    def empty_move(self):
        self.move(np.zeros((len(self._joints),)))

    def play_reset_actions(self):
        for v in self._reset_actions:
            self.move(v)

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

    def path_cost(self):
        return distance(self.get_path())

    def tip_path_cost(self):
        return distance(self.get_tip_path())

    def find_path_for_euler_angles(self, euler):
        return self.get_robot().get_path(
            position=self.get_target().get_position(),
            euler=euler,
            distance_threshold=self._threshold,
            max_time_ms=self.MAX_TIME_MS_DEFAULT,
            max_configs=self.MAX_CONFIGS_DEFAULT,
            algorithm=self.ALGORITHM_DEFAULT,
            trials=self.TRIALS_DEFAULT,
        )

    def find_optimal_path(self, step):
        roll, yaw, pitch = 0, 0, 0
        optimal_path = None
        optimal_cost = np.inf
        optimal_angles = None

        should_restart = False
        while roll < 2 * math.pi:
            while yaw < 2 * math.pi:
                while pitch < 2 * math.pi:

                    if should_restart:
                        self.clear_history()
                        self.restart_simulation()
                    path = None

                    try:
                        path = self.find_path_for_euler_angles([roll, yaw, pitch])
                        should_restart = True
                    except ConfigurationPathError:
                        should_restart = False

                    if path is not None:
                        done = False
                        self.update_history()

                        while not done:
                            done = path.step()
                            self.get_pyrep_instance().step()
                            self.update_history()

                        cost = self.path_cost()
                        if cost < optimal_cost and self.is_close():
                            optimal_cost = cost
                            optimal_path = path
                            optimal_angles = [roll, yaw, pitch]

                    print(roll, yaw, pitch, optimal_cost)
                    pitch += step

                yaw += step
                pitch = 0

            roll += step
            yaw = 0

        print(optimal_cost, optimal_angles)
        return path

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
