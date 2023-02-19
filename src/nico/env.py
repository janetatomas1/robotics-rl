

from gym.spaces import Box
from gym import Env
from pypot.pyrep import PyRepIO
from pyrep.objects import Shape
from nicomotion.Motion import Motion
import numpy as np
import math
from src.robot_env import RobotEnv


class NicoEnv(RobotEnv):
    def __init__(
        self,
        joints,
        scene,
        config,
        headless=False,
        **kwargs,
    ):
        self._io = PyRepIO(scene, headless=headless)
        super().__init__(
            scene=scene,
            pr=self._io.pyrep,
            **kwargs,
        )

        self._joints = joints
        self._timesteps_per_move = 10
        self._nreset_actions = 5

        pyrep_config = Motion.pyrepConfig()
        pyrep_config["shared_vrep_io"] = self._io
        pyrep_config["scene"] = self._scene

        self._robot = Motion(motorConfig=config, vrepConfig=pyrep_config, vrep=True)
        self._robot.startSimulation()

        self._low = np.array([math.degrees(self._robot.getAngleLowerLimit(j)) for j in self._joints])
        self._high = np.array([math.degrees(self._robot.getAngleUpperLimit(j)) for j in self._joints])

        self.observation_space = Box(
            low=np.concatenate([self._target_low, self._low]),
            high=np.concatenate([self._target_high, self._high])
        )
        self.action_space = Box(
            low=np.array([-10 for j in self._joints]),
            high=np.array([10 for j in self._joints]),
        )

        self._fraction_max_speed = 0.2

        self._tip_path = list()
        self._tip = self._io.get_object("r_indexfingers_x")

    def update_history(self):
        super().update_history()
        self._tip_path.append(self._tip.get_position())
        self._path.append(self.get_joint_values())

    def get_joint_values(self):
        return np.array([self._robot.getAngle(joint) for joint in self._joints])

    def get_state(self):
        return np.concatenate([self._target.get_position(), self.get_joint_values()])

    def is_close(self):
        return np.linalg.norm(
            np.array(self._target.get_position()) - np.array(self._io.get_object_position("r_indexfingers_x"))
        ) <= self._threshold

    def has_fallen(self):
        return min(
            [
                self._io.get_object_position("head_z")[2],
                self._io.get_object_position("head_y")[2],
            ]
        ) < self._threshold

    def is_done(self):
        return self.is_close() or self._episode_length <= self._steps or self.has_fallen()

    def reset(self):
        self.clear_history()
        self._io.restart_simulation()

        obs = self.observation_space.sample()
        self._target.set_position(obs[:3])

        for i in range(self._nreset_actions):
            action = self.action_space.sample()
            self.move(action)

        return self.get_state()

    def move(self, action):
        for joint, v in zip(self._joints, action):
            self._robot.changeAngle(jointName=joint, change=v, fractionMaxSpeed=self._fraction_max_speed)

        for i in range(self._timesteps_per_move):
            self._robot.nextSimulationStep()

    def distance(self):
        return np.linalg.norm(self._tip.get_position() - self.get_target().get_position())

    def clear_history(self):
        super().clear_history()
        self._tip_path.clear()

    def get_tip_path(self):
        return self._tip_path