
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


class PandaEnv(Env):
    INFO = {}

    def __init__(self, scene, threshold=0.3, joints=None, episode_length=100,
                 headless=False, reset_actions=10, log_dir=None, logger_class=None, reward_fn=sparse_reward):
        self.robot = None
        self.target = None
        self.reset_actions = reset_actions
        self.threshold = threshold
        self.headless = headless
        self.log_dir = log_dir
        self.episode_length = episode_length
        self.scene = scene
        self.reward = reward_fn
        self.steps = 0
        self.pyrep = PyRep()

        self.pyrep.launch(scene_file=self.scene, headless=headless)
        self.restart_simulation()
        self.joints = joints if joints is not None \
            else [i for i in range(len(self.robot.joints))]

        _, joint_intervals = self.robot.get_joint_intervals()
        self.low = np.array([joint_intervals[j][0] for j in self.joints])
        self.high = np.array([joint_intervals[j][1] for j in self.joints])

        self.observation_space = spaces.Box(
            low=np.concatenate([[0.8, -0.2, 0.5], self.low]),
            high=np.concatenate([[1.0, 0.2, 1.4], self.high]),
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
        self.steps += 1

        self.move(action)

        done = self.is_done()
        close = self.is_close()

        reward_kwargs = {
            "env": self,
            "done": done,
            "close": close,
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
