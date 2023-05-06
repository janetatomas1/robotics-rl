from gym import Env
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
import numpy as np

from roboticsrl.utils import distance
from roboticsrl.logger import JsonLogger


class RobotEnv(Env):
    """Generic class for RL environments supporting any robot form Pyrep"""
    def __init__(self,
                 scene,
                 target_low,
                 target_high,
                 robot_class=None,
                 log_file=None,
                 threshold=0.1,
                 episode_length=50,
                 headless=False,
                 reward_fn='const_sparse_reward',
                 dynamic_obstacles=False,
                 obstacles_low=None,
                 obstacles_high=None,
                 obstacles_state=None,
                 obstacles_type=PrimitiveShape.CUBOID,
                 success_reward=2,
                 collision_reward=-3,
                 failure_reward=-5,
                 step_failure_reward=-1,
                 ):
        """
        Parameters:
            scene: path to CoppeliaSim scene file
            target_low: lower boundary of the space containing the target
            target_high: upper boundary of the space containing the target
            robot_class: class of the robot from the Pyrep library
            threshold: maximal distance of the gripper from the target, to consider the episode suceesful
            episode_length: number of steps in an episode
            headless: headless mode
            reward_fn: a string representing reward function method
            dynamic_obstacles: boolean, detrmining if obstacles should be regenerated after each episode
            obstacles_low: lower boundary of the obstacle state space
            obstacles_high: upper boundary of the obstacle state space
            obstacles_state: obstacle state, used only with static obstacles
            obstacles_type: type of obstacles (SPHERE, CUBOID, CYLINDER,...)
            success_reward: reward returned in th case of a success,
            collision_reward: punishment reward, added in the case of a collision,
            failure_reward: reward returned after unsuccessful episode,
            step_failure_reward: reward returned after unsuccessful step,
        """
        self._threshold = threshold
        self._log_file = log_file
        self._episode_length = episode_length
        self._scene = scene
        self._steps = 0
        self._target_low = target_low
        self._target_high = target_high
        self._dynamic_obstacles = dynamic_obstacles
        self._obstacles_low = np.array([] if obstacles_low is None else np.concatenate(obstacles_low))
        self._obstacles_high = np.array([] if obstacles_high is None else np.concatenate(obstacles_high))
        self._obstacles_state = np.array([] if obstacles_state is None else np.concatenate(obstacles_state))
        self._obstacle_type = obstacles_type
        self._obstacle_color = [0, 0, 1]
        self._obstacles_number = 0

        if self._dynamic_obstacles:
            self._obstacles_number = len(obstacles_low)
        elif obstacles_state is not None:
            self._obstacles_number = len(obstacles_state)

        self._obstacles = list()

        self._rewards = list()
        self._reward_fn = getattr(self, reward_fn)

        self._pyrep = PyRep()
        self._pyrep.launch(scene_file=self._scene, headless=headless)
        self._pyrep.start()

        if robot_class is not None:
            self._robot = robot_class()
            self._initial_robot_state = self._robot.get_configuration_tree()

        self._target = Shape('target')
        self.create_obstacles()
        self._collision_count = 0

        for o in self._obstacles:
            o.set_collidable(True)

        self._logger = JsonLogger()

        if self._log_file is not None:
            self._logger.open(self._log_file)

        self._path = list()

        self._success_reward = success_reward
        self._collision_reward = collision_reward
        self._failure_reward = failure_reward
        self._step_failure_reward = step_failure_reward

    def reset_robot(self):
        """Resets the state of the robot to back to the initial"""
        self._pyrep.set_configuration_tree(self._initial_robot_state)

    def move(self, action):
        """Performs the action logic"""
        pass

    def const_sparse_reward(self):
        close = self.is_close()
        done = self.is_done()
        punishment = self._collision_reward if self.check_collision() else 0

        if close:
            return self._success_reward + punishment
        elif done:
            return self._failure_reward + punishment

        return self._step_failure_reward + punishment

    def joint_sparse_reward(self):
        done = self.is_done()
        close = self.is_close()
        punishment = self._collision_reward if self.check_collision() else 0

        if close:
            return self._success_reward - self.path_cost() + punishment
        elif done:
            return self._failure_reward + punishment

        return self._step_failure_reward + punishment

    def path_cost(self):
        return distance(self._path)

    def reset(self):
        """Performs the base reset logic"""
        self.reset_robot()
        self.reset_target()
        self.clear_history()

        if self._dynamic_obstacles:
            self.clear_obstacles()
            self.create_obstacles()

        return self.get_state()

    def render(self, mode="human"):
        self._pyrep.step_ui()

    def distance(self):
        """Should return the current distance to the target"""
        return 0

    def get_state(self):
        """Should return current state of the environment"""
        pass

    def get_robot(self):
        return self._robot

    def get_target(self):
        return self._target

    def is_close(self):
        "Should return whether the target has been reached"
        return bool(self.distance() <= self._threshold)

    def is_done(self):
        return self.is_close() or self._steps >= self._episode_length

    def info(self):
        return {}

    def clear_history(self):
        self._path.clear()
        self._rewards.clear()
        self._steps = 0
        self._collision_count = 0

    def update_history(self):
        self._rewards.append(self._reward_fn())

    def step(self, action):
        """Performs the step logic"""
        self._steps += 1
        self.update_history()

        self.move(action)
        done = self.is_done()
        reward = self._reward_fn()
        info = self.info()

        if done:
            self.update_history()

        if done and self._log_file is not None:
            self.save_history(
                rewards=self._rewards,
                **info,
            )

        return self.get_state(), reward, done, info

    def close(self):
        self._logger.close()
        self._pyrep.stop()
        self._pyrep.shutdown()

    def save_history(self, **kwargs):
        self._logger.save_history(**kwargs)

    def get_pyrep_instance(self):
        return self._pyrep

    def get_path(self):
        return self._path

    def get_rewards(self):
        return self._rewards

    def get_logger(self):
        return self._logger

    def get_obstacles(self):
        return self._obstacles

    def check_collision(self):
        """Return whether collision has occurred"""
        return False

    def clear_obstacles(self):
        for o in self._obstacles:
            o.remove()

    def create_obstacles(self):
        """Generates new obstacles"""
        if self._dynamic_obstacles:
            self._obstacles_state = np.random.uniform(self._obstacles_low, self._obstacles_high)

        self._obstacles = [
            Shape.create(
                type=self._obstacle_type,
                color=self._obstacle_color,
                size=self._obstacles_state[(6 * i):(6 * i + 3)].tolist(),
                position=self._obstacles_state[(6 * i + 3):(6 * i + 6)].tolist(),
                static=True,
                respondable=True,
            ) for i in range(self._obstacles_number)
        ]

    def reset_target(self):
        done = False

        while not done:
            position = np.random.uniform(self._target_low, self._target_high)
            self.get_target().set_position(position=position)
            self.get_pyrep_instance().step()
            done = not any([self._target.check_collision(o) for o in self._obstacles])

    def get_steps(self):
        return self._steps

    def get_collision_count(self):
        return self._collision_count
