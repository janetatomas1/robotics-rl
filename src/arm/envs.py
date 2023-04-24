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
from pyrep.const import ConfigurationPathAlgorithms

from src.robot_env import RobotEnv
from src.utils import distance, angle_distance


class ArmEnv(RobotEnv):
    def __init__(self,
                 reset_actions=5,
                 joints=None,
                 max_speed=1,
                 **kwargs):
        super().__init__(**kwargs)
        self._joints = joints if joints is not None else [i for i in range(len(self._robot.joints))]
        self._nreset_actions = reset_actions
        self._reset_actions = list()
        self._tip_path = list()
        self._quaternion_history = list()

        _, joint_intervals = self._robot.get_joint_intervals()

        self._low = np.concatenate([
            self._target_low,
            [joint_intervals[j][0] for j in self._joints],
            self._obstacles_low,
        ])

        self._high = np.concatenate([
            self._target_high,
            [joint_intervals[j][1] for j in self._joints],
            self._obstacles_high,
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

        self._starting_joint_positions = self.get_joint_values()
        self._reset_joint_positions = self.get_joint_values()
        self._control_loop = False

    def clear_history(self):
        super().clear_history()
        self._tip_path.clear()

    def set_reset_actions(self, actions):
        self._reset_actions = actions

    def update_history(self):
        super().update_history()
        self._path.append(self.get_joint_values())
        self._tip_path.append(self.get_robot().get_tip().get_position())
        self._quaternion_history.append(self.get_robot().get_tip().get_quaternion())

    def set_control_loop(self, value):
        self._control_loop = value
        self.get_robot().set_control_loop_enabled(self._control_loop)

    def move(self, action):
        for j, v in zip(self._joints, action):
            self._robot.joints[j].set_joint_target_velocity(v)

        self._pyrep.step()
        if self.check_collision():
            self._collision_count += 1

    def reset(self):
        super().reset()
        self._reset_actions.clear()
        self.get_pyrep_instance().step()

        for _ in range(self._nreset_actions):
            self._reset_actions.append(self.action_space.sample())

        self.play_reset_actions()
        self._reset_joint_positions = self.get_joint_values()
        self.empty_move()

        return self.get_state()

    def reset_robot(self):
        super().reset_robot()
        self.get_robot().set_joint_positions(self._starting_joint_positions)
        self.get_pyrep_instance().step_ui()

    def reset_joint_values(self):
        self.get_robot().set_joint_positions(self._reset_joint_positions)
        self.get_pyrep_instance().step_ui()

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
        if self._dynamic_obstacles:
            return np.concatenate([self._target.get_position(), self.get_joint_values(), self._obstacles_state])

        return np.concatenate([self._target.get_position(), self.get_joint_values()])

    def get_joints(self):
        return self._joints

    def is_close(self):
        return bool(self.distance() <= self._threshold)

    def info(self):
        return {
            'path_cost': self.path_cost(),
            'tip_path_cost': self.tip_path_cost(),
            'collision_count': self._collision_count,
            'close': int(self.is_close()),
        }

    def get_reset_actions(self):
        return self._reset_actions

    def get_tip_path(self):
        return self._tip_path

    def tip_path_cost(self):
        return distance(self.get_tip_path())

    def find_path_for_euler_angles(
            self,
            euler,
            trials=1,
            max_configs=1,
            max_time_ms=1,
            algorithm=ConfigurationPathAlgorithms.PRM,
    ):
        return self.get_robot().get_path(
            position=self.get_target().get_position(),
            euler=euler,
            trials=trials,
            max_configs=max_configs,
            max_time_ms=max_time_ms,
            algorithm=algorithm,
            distance_threshold=self._threshold,
        )

    def check_collision(self):
        return any([self.get_robot().check_arm_collision(o) for o in self.get_obstacles()])

    def get_starting_joint_values(self):
        return self._starting_joint_positions

    def set_reset_joint_values(self, values):
        self._reset_joint_positions = values

    def get_reset_joint_values(self):
        return self._reset_joint_positions

    def tip_boosted_sparse_reward(self):
        done = self.is_done()
        close = self.is_close()
        punishment = self._collision_reward if self.check_collision() else 0

        if close:
            return self._boosted_reward - self.tip_path_cost() + punishment
        elif done:
            return self._failure_reward + punishment

        return self._step_failure_reward + punishment

    def quaternion_angle_boosted_sparse_reward(self):
        done = self.is_done()
        close = self.is_close()
        punishment = self._collision_reward if self.check_collision() else 0

        if close:
            return self._boosted_reward - self.quaternion_angle_cost() + punishment
        elif done:
            return self._failure_reward + punishment

        return self._step_failure_reward + punishment

    def joint_path_angle_boosted_sparse_reward(self):
        done = self.is_done()
        close = self.is_close()
        punishment = self._collision_reward if self.check_collision() else 0

        if close:
            return self._boosted_reward - self.joint_path_angle_cost() + punishment
        elif done:
            return self._failure_reward + punishment

        return self._step_failure_reward + punishment

    def average_joint_angle_boosted_sparse_reward(self):
        done = self.is_done()
        close = self.is_close()
        punishment = self._collision_reward if self.check_collision() else 0

        if close:
            return self._boosted_reward - self.average_joint_angle_boosted_sparse_reward() + punishment
        elif done:
            return self._failure_reward + punishment

        return self._step_failure_reward + punishment

    def quaternion_angle_cost(self):
        return angle_distance(self._quaternion_history)

    def joint_path_angle_cost(self):
        return angle_distance(self.get_path())

    def average_joint_path_angle_cost(self):
        steps = self.get_steps()
        steps = steps if steps > 0 else 1
        return angle_distance(self.get_path()) / steps


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
