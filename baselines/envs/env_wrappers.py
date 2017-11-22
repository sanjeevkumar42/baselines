import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from gym.spaces.box import Box
from skimage import color
from dm_control import suite

from baselines.envs.dm_gym_wrapper import DMGymWrapper
from baselines.envs.torcs_env import TorcsEnv


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=1, repeat_overlapping=True):
        super(ActionRepeatWrapper, self).__init__(env)
        space = env.observation_space
        self.overlapping = repeat_overlapping
        state_space_shape = space.shape[:-1] + (space.shape[-1] * action_repeat,)
        self.observation_space = Box(0.0, 1.0, state_space_shape)
        self.obs_history = []
        self.action_repeat = action_repeat

    def _step(self, action):
        ob_state = []
        total_reward = 0
        done = False
        no_steps = 1 if self.overlapping else self.action_repeat
        for i in range(no_steps):
            state, reward, terminal, info = self.env.step(action)
            ob_state.append(state)
            total_reward += reward
            done = done or terminal
        self.obs_history = self.obs_history + ob_state
        self.obs_history = self.obs_history[-self.action_repeat:]
        return np.concatenate(self.obs_history, axis=-1), total_reward, done, info

    def _reset(self):
        ob = self.env.reset()
        self.obs_history = [ob] * self.action_repeat
        return np.concatenate(self.obs_history, axis=-1)


class VisualEnv(gym.Wrapper):
    def __init__(self, env, visible=False, init_width=64, init_height=64, go_fast=False, grayscale=False):
        super(VisualEnv, self).__init__(env)
        self.visible = visible
        self.init_width = init_width
        self.init_height = init_height
        self.go_fast = go_fast
        self.grayscale = grayscale
        if self.grayscale:
            channels = 1
        else:
            channels = 3
        self.observation_space = spaces.Box(low=0, high=255, shape=(init_height, init_width, channels))
        if hasattr(self.env.env, '_get_viewer'):
            self.env.env._get_viewer(visible, init_width, init_height, go_fast)

    def _postprocess_image(self, img_data):
        if self.grayscale:
            gray_img = color.rgb2gray(img_data)
            gray_img = np.expand_dims(gray_img, -1)
            return gray_img
        else:
            return img_data * 1. / 255

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        image_data = self.env.render(mode='rgb_array')
        image_data = self._postprocess_image(image_data)
        return image_data, reward, done, info

    def _reset(self, **kwargs):
        self.env.reset()
        image_data = self.env.render(mode='rgb_array')
        image_data = self._postprocess_image(image_data)
        return image_data


class TorcsRescale(gym.Wrapper):
    def __init__(self, env=None, width=84, height=84, **kwargs):
        super(TorcsRescale, self).__init__(env)
        self.height = height
        self.width = width
        self.observation_space = Box(0.0, 1.0, [height, width, 1])

    def _step(self, action):
        state, reward, terminal, info = self.env.step(action)
        return self._process_frame_torcs(state), reward, terminal, info

    def _reset(self):
        ob = self.env.reset()
        return self._process_frame_torcs(ob)

    def _process_frame_torcs(self, frame):
        frame = cv2.resize(frame, (self.width, self.height))
        frame = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame * (1.0 / 255.0)
        frame = np.reshape(frame, self.observation_space.shape)
        return frame


def make_visual_env(env_id, action_repeat=3, grayscale=False, repeat_overlapping=False):
    if 'Torcs-v0' == env_id:
        env = TorcsEnv(10, 84, 84, frame_skip=1)
        env = TorcsRescale(env)
        env = ActionRepeatWrapper(env, action_repeat=action_repeat, repeat_overlapping=repeat_overlapping)
    else:
        # try to make dm_control env first
        env = gym.make(env_id)
        env = VisualEnv(env, grayscale=grayscale)
        env = ActionRepeatWrapper(env, action_repeat=action_repeat, repeat_overlapping=repeat_overlapping)
    return env


def make_dm_env(env_name):
    domain_name, task_name = env_name.split('-')
    env = suite.load(domain_name=domain_name, task_name=task_name)
    env = DMGymWrapper(env)
    return env


def make_env(env_id):
    try:
        env = make_dm_env(env_id)
    except:
        env = gym.make(env_id)

    return env


if __name__ == '__main__':
    env = make_visual_env('HalfCheetah-v1', action_repeat=4, grayscale=True, repeat_overlapping=True)
    ob = env.reset()
    while True:
        ob = env.step(env.action_space.sample())
        cv2.imwrite('/data/test.png', ob[0][:, :, 2::-1] * 255)
        plt.figure(1), plt.imshow(ob[0][:, :, 0])
        plt.figure(2), plt.imshow(ob[0][:, :, 1])
        plt.figure(3), plt.imshow(ob[0][:, :, 2])
        plt.figure(4), plt.imshow(ob[0][:, :, 3])
        plt.show()
