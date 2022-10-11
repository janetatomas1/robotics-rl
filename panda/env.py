
from gym import spaces, Env
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np


class PandaEnv(Env):
    def __init__(self, scene, threshold=0.3, joints=None):
        self.threshold = threshold
        self.scene = scene
        self.pyrep = PyRep()
        self.pyrep.launch(scene_file=self.scene)
        self.pyrep.start()
        self.pyrep.step()
        self.robot = Panda()
        self.robot.set_control_loop_enabled(False)
        self.robot.set_motor_locked_at_zero_velocity(True)
        self.joints = joints if joints is not None \
            else [i for i in range(len(self.robot.joints))]
        self.target = Shape('target')
        self.observation_space = spaces.Box(
            low=np.array(2 * [-3., -3., 0.]),
            high=np.array(2 * [3., 3., 2.]),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=np.array([-2 for j in self.joints]),
            high=np.array([2 for j in self.joints])
        )

    def reset(self):
        return self._get_state()

    def render(self, mode="human"):
        pass

    def _get_state(self):
        return np.concatenate([self.robot.get_tip().get_position(), self.target.get_position()])

    def _is_close(self):
        return bool(np.linalg.norm(self.target.get_position() - self.robot.get_tip().get_position()) <= self.threshold)

    def _reward(self):
        return -np.linalg.norm(self.target.get_position() - self.robot.get_tip().get_position())

    def info(self):
        return {}

    def step(self, action):
        self.robot.set_joint_target_velocities(action)
        self.pyrep.step()

        return self._get_state(), self._reward(), self._is_close(), self.info()
