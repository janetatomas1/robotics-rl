import multiprocessing.process
import pathlib
import sys
import time
from multiprocessing import Process
import json
import numpy as np
import random
from pyrep.errors import ConfigurationPathError
from .envs import (
    PandaEnv,
)
import math
import threading

from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory


n = 1000
m = 5
scene = str(pathlib.Path(pathlib.Path(__file__).parent.parent.parent, 'scenes', 'scene_panda.ttt'))
target_low, target_high = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
step = math.radians(15)
alphabet = 'abcdefghijklmnopqrstuvxyz0123456789'


def random_id():
    return ''.join(random.sample(alphabet, 10))


def generate_positions():
    env = PandaEnv(scene=scene, target_low=target_low, target_high=target_high, max_speed=0.5)
    states = list()
    for i in range(n):
        env.set_control_loop(False)
        env.empty_move()
        env.reset()
        states.append({
            'actions': np.array(env._reset_actions).tolist(),
            'target_pos': list((env.get_target().get_position())),
            'id_': random_id(),
            'joints': env.get_joint_values().tolist(),
        })
        print(i)
    with open('/opt/positions/positions.json', 'w') as file:
        json.dump(states, file, indent=' ')
    env.close()
    sys.exit(1)


def check_angle(x):
    return all([a + step > math.pi * 2 for a in x])


def find_path_process(id_):
    env = PandaEnv(scene=scene, target_low=target_low, target_high=target_high)
    file = open('/opt/results/{}.json'.format(id_))
    state = json.load(file)
    file.close()

    actions = state['actions']
    roll, yaw, pitch = state['euler']
    distance = state['distance']
    target_pos = state['target_pos']
    joints = state['joints']


    env._control_loop = True

    try:
        shm = SharedMemory(size=3, name='aaa', create=True)
    except:
        shm = SharedMemory(size=3, name='aaa')

    def control():
        while True:
            time.sleep(1)
            shm.buf[0] += 1
            if shm.buf[0] > 10:
                env.close()
                sys.exit(1)
            print(shm.buf[0], env.get_pyrep_instance().running)

    t = threading.Thread(target=control)
    t.start()
    env.set_control_loop(True)

    should_restart = True
    env.get_pyrep_instance().step()
    env.get_target().set_position(target_pos)
    while roll < 2 * math.pi:
        while yaw < 2 * math.pi:
            while pitch < 2 * math.pi:
                if should_restart:
                    env.clear_history()
                    env.set_reset_joint_values(joints)
                    env.reset_joint_values()
                    env.get_pyrep_instance().step_ui()
                path = None

                try:
                    path = env.find_path_for_euler_angles(euler=[roll, yaw, pitch])
                    should_restart = True
                except ConfigurationPathError:
                    should_restart = False

                if path is not None:
                    done = False
                    env.update_history()

                    while not done:
                        done = path.step()
                        env.get_pyrep_instance().step()
                        env.update_history()

                    cost = env.path_cost()
                    print("path", env.is_close())
                    if cost < state['distance'] and env.is_close():
                        state['distance'] = cost
                        state['path'] = np.array(env.get_path()).tolist()
                        state['tip_distance'] = env.tip_path_cost()
                        state['tip_path'] = np.array(env.get_tip_path()).tolist()
                        state['optimal_euler'] = [roll, yaw, pitch]
                state['euler'] = [roll, yaw, pitch]
                state['finished'] = check_angle([roll, yaw, pitch])
                file = open('/opt/results/{}.json'.format(id_), 'w')
                json.dump(state, file, indent='\t')
                file.close()
                print(shm.buf[0], shm.buf[1], roll, yaw, pitch, state['finished'], env.distance(), id_)

                if state['finished']:
                    shm.buf[1] = 1
                    while True:
                        pass

                shm.buf[0] = 1
                file.close()
                pitch += step
            yaw += step
            pitch = 0
        yaw = 0
        roll += step


def find_optimal_path(actions, target_pos, id_, joints):
    euler = [0, 0, 0]

    with open('/opt/results/{}.json'.format(id_), 'w') as file:
        json.dump({
            'actions': actions,
            'target_pos': target_pos,
            'path': [],
            'distance': 1000000,
            'tip_distance': 1000000,
            'euler': euler,
            'tip_path': [],
            'finished': False,
            'joints': joints,
        }, file, indent='\t')

    try:
        shm = SharedMemory(size=3, name='aaa', create=True)
    except:
        shm = SharedMemory(size=3, name='aaa')
    shm.buf[0] = 0
    shm.buf[1] = 0

    while True:
        if shm.buf[0] == 0:
            proc = Process(target=find_path_process, args=(id_,))
            shm.buf[0] = 1
            time.sleep(2)
            proc.start()

        if shm.buf[0] > 10:
            proc.kill()
            shm.buf[0] = 0

            if shm.buf[1] == 1:
                shm.buf[1] = 0
                shm.unlink()
                return
        time.sleep(1)

        print('alive', proc.is_alive(), shm.buf[0])


def train():
    # generate_positions()
    with open('/opt/positions/positions.json') as file:
        states = json.load(file)

    for s in states:
        find_optimal_path(**s)
