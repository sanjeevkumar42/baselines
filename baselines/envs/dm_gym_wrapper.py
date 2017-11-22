import gym
from dm_control.rl.environment import StepType
from gym import spaces

import numpy as np

from gym.utils import seeding


class DMGymWrapper(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, dm_env, visual=False):
        self.dm_env = dm_env
        self.visual = visual
        action_spec = dm_env.action_spec()
        obs_spec = dm_env.observation_spec()
        self.action_space = spaces.Box(action_spec.minimum, action_spec.maximum)
        if visual:
            self.observation_space = spaces.Box(0, 255, shape=())
        else:
            nb_obs = 0
            for v in obs_spec.values():
                nb_obs += v.shape[0] if len(v.shape) > 0 else 1

            self.observation_space = spaces.Box(-np.inf, np.inf, nb_obs)

        self.viewer = None

    def _step(self, action):
        obs = self.dm_env.step(action)
        done = obs.step_type == StepType.LAST
        info = {'raw_dm': obs}
        return np.hstack(obs.observation.values()), obs.reward, done, info

    def _reset(self):
        obs = self.dm_env.reset()
        return np.hstack(obs.observation.values())

    def _render(self, mode='human', close=False):
        image = self.dm_env.physics.render()
        if mode == 'rgb_array':
            return image
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(image)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
