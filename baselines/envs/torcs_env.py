import os
import random
from sysv_ipc import SharedMemory
from threading import Thread

import cv2
import sys
from gym.envs.registration import EnvSpec
from subprocess import Popen

import subprocess

import gym
import math

import time
import numpy as np
from gym import spaces
from gym.utils import seeding

from baselines.envs import xserver_util, snakeoil
from baselines.envs.logger import Logger


class ProcessLogPoller(Thread):
    def __init__(self, logfile):
        Thread.__init__(self)
        self.logfile = os.path.join(logfile, 'torcs.log')
        self.process = None

    def set_process(self, process):
        self.process = process

    def run(self):
        while True:
            if self.process:
                line = self.process.stdout.readline()
                if line:
                    with open(self.logfile, 'a') as fw:
                        fw.write('Pid:{}, output:{}\n'.format(self.process.pid, line))
            else:
                time.sleep(1)


env_instances = 0
max_instances = 100


class TorcsEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    min_steps = 2000
    min_speed = 5  # speed should not be less than 5km/h after min_steps

    ACTIONS = [('steer', -1), ('steer', 0), ('steer', 1), ('accel', 0), ('accel', 0.5), ('accel', 1), ('brake', 0),
               ('brake', 0.5), ('brake', 1)]
    spec = EnvSpec('Torcs-v0', max_episode_steps=10000, reward_threshold=20000.0)

    class KEY_MAP:
        ENTER = 'KP_Enter'
        UP = 'KP_Up'

    LAUNCH_SEQ = [KEY_MAP.ENTER, KEY_MAP.ENTER, KEY_MAP.UP, KEY_MAP.UP, KEY_MAP.ENTER, KEY_MAP.ENTER]

    def __init__(self, env_id, width=64, height=64, frame_skip=(2, 5), torcs_dir='/usr/local',
                 logdir='/data/logs', **kwargs):
        # global env_instances
        # env_instances = env_instances % max_instances + 1
        self.env_id = env_id
        self.port = 9505 + self.env_id
        self.disp_name = ':{}'.format(20 + self.env_id)
        self.screen_w, self.screen_h = width, height
        self.torcs_dir = torcs_dir
        self.viewer = None
        self.time_step = 0
        self.client = None
        self.torcs_process = None
        self.shared_memory = None
        self.frameskip = frame_skip
        # self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.action_space = spaces.Box(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_h, self.screen_w, 3))
        self.reward_range = (-1, 50)
        self._seed()
        self.logdir = os.path.join(logdir, 'torcs-{}'.format(env_id))
        self.logger = Logger(self.env_id, self.logdir)
        self.dist_raced = 0.0
        self.done = False

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def __start_torcs(self):
        if self.torcs_process:
            subprocess.call('rm -rf /tmp/.X*', shell=True)
            self.torcs_process.kill()
            self.disp_process.kill()
        else:
            self.torcs_logpoller = ProcessLogPoller(self.logdir)
            self.torcs_logpoller.start()

        color_depth = 24
        screen_res = '{}x{}x{}'.format(self.screen_w, self.screen_h, color_depth)
        self.disp_process = subprocess.Popen(['Xvfb', self.disp_name, '-ac', '-screen', '0', screen_res],
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        self.logger.info('Xvfb pid:{}, display:{}', self.disp_process.pid, self.disp_name)
        torcs_bin = os.path.join(self.torcs_dir, 'lib/torcs/torcs-bin')
        torcs_lib = os.path.join(self.torcs_dir, 'lib/torcs/lib')
        torcs_data = os.path.join(self.torcs_dir, 'share/games/torcs/')
        self.torcs_process = Popen(
            [torcs_bin, '-port', str(self.port), '-nofuel', '-nodamage', '-nolaptime', '-cmdFreq', '25', '-h',
             str(self.screen_h), '-w', str(self.screen_w)],
            env={'LD_LIBRARY_PATH': torcs_lib, 'DISPLAY': self.disp_name},
            cwd=torcs_data, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.torcs_logpoller.set_process(self.torcs_process)
        for cmd in self.LAUNCH_SEQ:
            time.sleep(0.8)
            subprocess.call(['xdotool', 'key', cmd], env={'DISPLAY': self.disp_name})
        self.shared_memory = SharedMemory(self.port)
        self.logger.info('Started torcs pid:{}, port:{}', self.torcs_process.pid, self.port)

    def __to_torcs_action(self, action):
        torcs_actions = {a[0]: 0.0 for a in self.ACTIONS}
        torcs_actions[self.ACTIONS[action][0]] = self.ACTIONS[action][1]
        return torcs_actions

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}
        torcs_action.update({'accel': (1. + u[1]) / 2})
        torcs_action.update({'brake': (1. + u[2]) / 2})
        return torcs_action

    def _step(self, action):
        if self.done:
            return self.image, 0.0, self.done, self.info

        torcs_action = self.agent_to_torcs(action)
        # print('Action:{}'.format(torcs_action))

        self.client.R.d.update(torcs_action)
        self.client.R.d['gear'] = 1
        if self.client.S.d['speedX'] > 50:
            self.client.R.d['gear'] = 2

        if self.client.S.d['speedX'] > 80:
            self.client.R.d['gear'] = 3

        reward = 0.0
        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])

        self.logger.info('Sending command to server:{}', self.client.R.d)
        self.logger.info('Server state:{}', self.client.S.d)
        for i in range(num_steps):
            self.client.respond_to_server()
            self.client.get_servers_input()
            speed = self.client.S.d['speedX']
            angle = self.client.S.d['angle']
            dist_raced = self.client.S.d['distRaced']

            # reward += dist_raced - self.dist_raced + speed * math.cos(angle) / 100.0
            # reward += speed*math.cos(angle)
            reward += speed * math.cos(angle) - np.abs(speed * math.sin(angle)) - speed * abs(
                self.client.S.d['trackPos'])
            self.dist_raced = dist_raced

        long_speed = speed * math.cos(angle)  # speed along x-axis

        track = np.array(self.client.S.d['track'])

        if track.min() < 0 or np.cos(angle) < 0 or self.time_step > self.min_steps and long_speed < 5:
            reward = -100
            done = True
            print(self.client.S.d)
            self.logger.debug('Terminal state!! track:{}, angle:{}, speed:{}, steps:{}', track, angle, long_speed,
                              self.time_step)
        else:
            done = False

        self.image = xserver_util.get_screen_shm(self.shared_memory, self.screen_w, self.screen_h)
        self.info = {'state': dict(self.client.S.d)}
        # print(self.client.S.d)
        if self.time_step % 50 == 0 or done:
            filename = os.path.join(self.logdir, '{}_{}.png'.format(self.torcs_process.pid, self.time_step))
            cv2.imwrite(filename, self.image)
        self.time_step += 1
        self.done = done
        return self.image, reward, done, self.info

    def _reset(self):
        self.logger.info('Resetting torcs!!')
        while True:
            try:
                self.__start_torcs()
                self.client = snakeoil.Client(p=self.port)
                self.client.get_servers_input()
                break
            except Exception as e:
                self.logger.info("Failed to connect to torcs. Will try again.")
                continue
        self.dist_raced = 0.0
        self.image = xserver_util.get_screen_shm(self.shared_memory, self.screen_w, self.screen_h)
        self.info = {'state': dict(self.client.S.d)}
        self.logger.info('Successfully reset torcs after timesteps:{}', self.time_step)
        self.time_step = 0
        self.done = False
        return self.image

    def _render(self, mode='human', close=False):
        if close:
            self.torcs_process.kill()
            self.disp_process.kill()
        if mode == 'rgb_array':
            return self.image
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.image)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class SensorBasedTorcsEnv(TorcsEnv):
    def __init__(self, *args, **kwargs):
        super(SensorBasedTorcsEnv, self).__init__(*args, **kwargs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,))

    def _step(self, action):
        ob, reward, done, info = super(SensorBasedTorcsEnv, self)._step(action)
        state = self._sensor_ob_to_state(info)
        return state, reward, done, info

    def _sensor_ob_to_state(self, info):
        sensor_obs = info['state']
        angle = np.array(sensor_obs['angle'], dtype=np.float32) / math.pi
        # focus = np.array(sensor_obs['focus'], dtype=np.float32) / 200.0
        speedx = np.array(sensor_obs['speedX'], dtype=np.float32) / 300.0
        speedy = np.array(sensor_obs['speedY'], dtype=np.float32) / 300.0
        speedz = np.array(sensor_obs['speedZ'], dtype=np.float32) / 300.0
        track = np.array(sensor_obs['track'], dtype=np.float32) / 200.0
        wheel_spin = np.array(sensor_obs['wheelSpinVel'], dtype=np.float32) / 100.0
        rpm = np.array(sensor_obs['rpm'], dtype=np.float32) / 10000
        track_pos = np.array(sensor_obs['trackPos'], dtype=np.float32)
        state = np.hstack((angle, track, track_pos, speedx, speedy, speedz, wheel_spin, rpm))
        return state

    def _reset(self):
        super(SensorBasedTorcsEnv, self)._reset()
        state = self._sensor_ob_to_state(self.info)
        return state


R = {'accel': 0.0, 'gear': 1, 'steer': 0}


def drive_example(S):
    '''This is only an example. It will get around the track but the
    correct thing to do is write your own `drive()` function.'''
    target_speed = 100

    # Steer To Corner
    R['steer'] = S['angle'] * 10 / math.pi
    # Steer To Center
    R['steer'] -= S['trackPos'] * .10

    # Throttle Control
    if S['speedX'] < target_speed - (R['steer'] * 50):
        R['accel'] += .01
    else:
        R['accel'] -= .01
    if S['speedX'] < 10:
        R['accel'] += 1 / (S['speedX'] + .1)

    # Traction Control System
    if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) -
            (S['wheelSpinVel'][0] + S['wheelSpinVel'][1]) > 5):
        R['accel'] -= .2
    # R['accel'] = 1.0
    # R['steer'] = 0.0

    # Automatic Transmission
    R['gear'] = 1
    if S['speedX'] > 50:
        R['gear'] = 2
    if S['speedX'] > 80:
        R['gear'] = 3
    if S['speedX'] > 110:
        R['gear'] = 4
    if S['speedX'] > 140:
        R['gear'] = 5
    if S['speedX'] > 170:
        R['gear'] = 6

    if R['accel'] < 0.25:
        accel = 3
    elif R['accel'] > 0.25 and R['accel'] < 0.75:
        accel = 4
    else:
        accel = 5

    if R['steer'] < -0.75:
        steer = 0
    elif R['steer'] > -0.75 and R['steer'] < 0.25:
        steer = 1
    else:
        steer = 2
    steer_noise = np.random.normal(0, 0.5)
    accel_noise = np.random.normal(0, 0.5)
    return [R['steer'] + steer_noise, (R['accel'] + accel_noise) * 2 - 1, -1]


if __name__ == '__main__':
    # env = TorcsEnv(35, frame_skip=1, height=120, width=160)
    env = SensorBasedTorcsEnv(13, frame_skip=1, height=128, width=128)
    env.reset()
    total_reward = 0
    info = {}
    for i in range(100000):
        if info:
            a = drive_example(info['state'])
        else:
            a = [0, 1, 0]
        ob, reward, done, info = env.step(a)
        total_reward += reward
        if done:
            print(total_reward)
            total_reward = 0
            env.reset()
        # env.render()
        # if i % 50 == 0:
        #     print('Resetting! Done.')
        #     a.reset()
        print('step:{}, reward:{} '.format(i, reward))
        # if i % 10 == 0:
        #     env.render()
        # cv2.imwrite('/data/out/torcs_vae/images/{}.png'.format(i), env.render('rgb_array'))
    sys.exit(0)
