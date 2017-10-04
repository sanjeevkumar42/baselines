from queue import Queue

import gym
import time

import math
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from gym.utils import seeding
from scipy.misc import imresize


class Maze2D(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    spec = EnvSpec('Maze2D-v0', max_episode_steps=500, reward_threshold=500.0)

    class MAZE_STATE:
        FREE_SPACE = 0
        OBSTACLE = 1
        TARGET = 2
        VISITED = 3
        CURRRENT_POS = 4

    def __init__(self, maze_shape=(100, 100), obstacles=None, targets=None, start_pos=None, obs_area=5,
                 frame_skip=1, vis_shape=(500, 500), obs_complete=True):
        self.maze_shape = maze_shape
        self.viewer = None
        self.total_steps = 0
        self.obstacles = obstacles
        self.targets = targets
        self.frameskip = frame_skip
        self.obs_area = obs_area
        self.obs_complete = obs_complete
        self.start_pos = start_pos
        self.action_space = spaces.Box(np.array([-1.0]), np.array([1.0]))
        if self.obs_complete:
            self.observation_space = spaces.Box(0, 4, self.maze_shape[0] * self.maze_shape[1])
        else:
            self.observation_space = spaces.Box(0, 4, ((2 * self.obs_area + 1) * (2 * self.obs_area + 1)))
        self.reward_range = (-10, 500)

        self.vis_shape = vis_shape
        self.cmap = color_map(range(5))

        self._seed()
        self.done = False

        self.laststeps = []

    def hit_shape(self, x1, y1, x2, y2, shapes):
        if x2 < 0 or x2 >= self.maze_shape[0] or y2 < 0 or y2 >= self.maze_shape[1]:
            return True
        for o in shapes:
            if o[0] == 1:
                ot, cx, cy, r = o

                d = abs((x2 - x1) * cx + (y1 - y2) * cy + (x1 - x2) * y1 + (y2 - y1) * x1) / math.sqrt(
                    (x2 - x1) ** 2 + (y2 - y1) ** 2)
                if d <= r:
                    return True

            elif o[0] == 2:
                st, sx, sy, sh, sw = o
                box = np.array([[sx - sh, sx + sh], [sy - sw, sy + sw]])
                box = np.clip(box, [0, 0], [self.grid.shape[0] - 1, self.grid.shape[1] - 1])
                # l1 = ((x1, y1), (x2, y2))
                if box[0][0] <= x2 <= box[0][1] and box[1][0] <= y2 <= box[1][1]:
                    return True
        return False

    def intersect(self, l1, l2):
        (x00, y00), (x01, y01) = l1
        (x10, y10), (x11, y11) = l2
        d = x11 * y01 - x01 * y11
        if d != 0:
            s = (1 / d)((x00 - x10) * y01 - (y00 - y10) * x01)
            t = (1 / d) - (-(x00 - x10) * y11 + (y00 - y10) * x11)
            if 0 <= s <= 1 and 0 <= t <= 1:
                return True
            else:
                return False
        else:
            return False

    def obs_grid(self, x, y):
        if self.obs_complete:
            return self.grid
        else:
            offset = self.obs_area
            x = int(x)
            y = int(y)
            obs_grid = np.zeros((2 * offset + 1, 2 * offset + 1))
            for i in range(-offset, offset + 1):
                for j in range(-offset, offset + 1):
                    if x + i < 0 or y + j < 0:
                        obs_grid[i + offset][j + offset] = Maze2D.MAZE_STATE.OBSTACLE
                    elif x + i >= self.maze_shape[0] or y + j >= self.maze_shape[1]:
                        obs_grid[i + offset][j + offset] = Maze2D.MAZE_STATE.OBSTACLE
                    else:
                        obs_grid[i + offset][j + offset] = self.grid[x + i][y + j]

            return obs_grid

    def _step(self, action):
        """
        Args:
            action: Angle [-1,1] -> [-pi/2, pi/2]
             R - distance to move. [-1, 1]

        """

        theta = action[0]
        r = 1.0
        sina = math.sin(math.pi * theta)
        cosa = math.cos(math.pi * theta)

        new_x, new_y = self.lastx + r * cosa, self.lasty + r * sina
        done = False
        if self.total_steps > self.maze_shape[0] * self.maze_shape[1]:
            reward = -10.0
            done = True
        elif new_x < 0 or new_x >= self.maze_shape[0] or new_y < 0 or new_y >= self.maze_shape[1] or self.grid[
            int(new_x), int(new_y)] == Maze2D.MAZE_STATE.OBSTACLE:
            reward = -10.0
            done = True
        elif self.grid[int(new_x), int(new_y)] == Maze2D.MAZE_STATE.TARGET:
            reward = 10.0
            done = True
        else:
            reward = 0.0
            self.grid[int(new_x), int(new_y)] = Maze2D.MAZE_STATE.CURRRENT_POS
            self.grid[int(self.lastx), int(self.lasty)] = Maze2D.MAZE_STATE.VISITED
            self.lastx = new_x
            self.lasty = new_y
        # print('New pos:{}, pos:{}, r:{}, angle:{}'.format(new_x, new_y, r, math.pi * theta))
        # if self.hit_shape(self.lastx, self.lasty, new_x, new_y, self.obstacles):
        #     reward = -10.0
        #     self.grid[int(self.lastx), int(self.lasty)] = Maze2D.MAZE_STATE.VISITED
        #     self.lastx = new_x
        #     self.lasty = new_y
        #     done = True
        # elif self.hit_shape(self.lastx, self.lasty, new_x, new_y, self.targets):
        #     reward = 10
        #     done = True
        #     self.grid[int(self.lastx), int(self.lasty)] = Maze2D.MAZE_STATE.VISITED
        #     self.lastx = new_x
        #     self.lasty = new_y
        # else:
        #     if len(self.laststeps) == 20:
        #         self.laststeps.pop(0)
        #     self.laststeps.append((new_x, new_y))
        #
        #     reward = 0.0
        #     self.total_steps += 1
        #     if self.grid[int(new_x), int(new_y)] == Maze2D.MAZE_STATE.OBSTACLE:
        #         reward = -10
        #         pass
        #     else:
        #         self.grid[int(new_x), int(new_y)] = Maze2D.MAZE_STATE.CURRRENT_POS
        #         self.grid[int(self.lastx), int(self.lasty)] = Maze2D.MAZE_STATE.VISITED
        #         self.lastx = new_x
        #         self.lasty = new_y
        # print('action :{}, position: ({},{})'.format(action, self.lastx, self.lasty))

        self.total_steps += 1
        return self.obs_grid(self.lastx, self.lasty).flatten(), reward, done, {}

    def _reset(self):
        self.grid = self.__build_maze()
        self.total_steps = 0

        self.done = False
        return self.obs_grid(self.lastx, self.lasty).flatten()

    def __build_maze(self):
        grid = np.full(self.maze_shape, Maze2D.MAZE_STATE.FREE_SPACE)
        for o in self.obstacles:
            fill_shape(grid, o, value=Maze2D.MAZE_STATE.OBSTACLE)

        idx = np.where(grid == Maze2D.MAZE_STATE.FREE_SPACE)
        idx = list(zip(idx[0], idx[1]))

        if self.start_pos:
            self.lastx, self.lasty = self.start_pos
        else:
            sel = np.random.randint(len(idx))
            self.lastx, self.lasty = idx.pop(sel)

        grid[self.lastx, self.lasty] = Maze2D.MAZE_STATE.VISITED
        self.laststeps = []
        self.laststeps.append((self.lastx, self.lasty))

        self.targets = self.find_empty_targets(grid)
        # print('Targets:{}'.format(self.targets))
        for t in self.targets:
            fill_shape(grid, t, value=Maze2D.MAZE_STATE.TARGET)

        return grid

    def find_empty_targets(self, grid, no_targets=1, target_offset=(1, 1)):
        empty_targets = []
        x_offset, y_offset = target_offset
        for i in range(x_offset, grid.shape[0] - x_offset):
            for j in range(y_offset, grid.shape[1] - y_offset):
                target = grid[i - x_offset:i + x_offset + 1, j - y_offset:j + y_offset + 1]
                if np.all(target == 0):
                    empty_targets.append((2, i, j, x_offset, y_offset))

        np.random.shuffle(empty_targets)
        targets = empty_targets[:no_targets]
        return targets

    def _render(self, mode='human', close=False):
        if self.obs_complete:
            vis_img = self.create_rgb_image([self.grid])
        else:
            vis_img = self.create_rgb_image([self.grid, self.obs_grid(self.lastx, self.lasty)])
        if mode == 'rgb_array':
            return vis_img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(vis_img)

    def create_rgb_image(self, images):
        arr = np.zeros((self.vis_shape[0], len(images) * self.vis_shape[1], 3), dtype=np.uint8)
        for i, img in enumerate(images):
            rgb_img = heapmap(img, self.cmap)
            rgb_img = np.transpose(rgb_img, (1, 0, 2))
            arr[:, i * self.vis_shape[1]:(i + 1) * self.vis_shape[0]] = imresize(rgb_img, self.vis_shape,
                                                                                 interp='nearest')
        return arr

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def fill_shape(grid, shape_info, value=1):
    if shape_info[0] == 1:
        st, sx, sy, sr = shape_info
        fill_circle(grid, (sx, sy), sr, value)
    elif shape_info[0] == 2:
        st, sx, sy, sh, sw = shape_info
        fill_rectangle(grid, (sx, sy), (sh, sw), value)


def fill_rectangle(grid, center, size, value=1):
    box = np.array([[center[0] - size[0], center[0] + size[0] + 1], [center[1] - size[1], center[1] + size[1] + 1]])
    box = np.clip(box, [0, 0], [grid.shape[0], grid.shape[1]])
    grid[box[0][0]:box[0][1], box[1][0]:box[1][1]] = value


def fill_circle(grid, center, rad, value=1):
    xx = np.arange(grid.shape[0])
    yy = np.arange(grid.shape[1])
    inside = (xx[:, None] - center[0]) ** 2 + (yy[None, :] - center[1]) ** 2 <= (rad ** 2)
    grid[inside] = value


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def color_map(values):
    cmap = {}
    minv = min(values)
    maxv = max(values)
    for value in values:
        cmap[value] = rgb(minv, maxv, value)
    return cmap


def heapmap(array, cmap):
    rgb_image = np.zeros(array.shape + (3,), dtype=np.uint8)
    for v, color_value in cmap.items():
        rgb_image[array == v, :] = color_value
    return rgb_image


if __name__ == '__main__':
    # obst = [(2, 30, 30, 5, 10), (2, 30, 60, 5, 10), (2, 65, 65, 35, 5),
    #         (1, 30, 75, 5), (1, 5, 45, 5), (2, 65, 20, 5, 20)]
    # env = Maze2D(frame_skip=1, obstacles=obst)
    obst = [(2, 22, 6, 2, 6), (2, 10, 10, 2, 3), (2, 18, 20, 10, 2),
            (1, 2, 14, 2), (1, 5, 28, 1)]
    env = Maze2D(maze_shape=(32, 32), obstacles=obst, targets=[(1, 30, 30, 2)], obs_area=8, obs_complete=False)

    env.reset()
    for i in range(1000):
        a = env.action_space.sample()
        if i < 39:
            a = [0]
        elif i < 60:
            a = [0]
        elif i < 120:
            a = [1.0]
        else:
            a = [0]
        ob, reward, done, info = env.step(a)
        print(a, reward, done)
        # if i % 10 == 0:
        env.render()
        if done:
            env.reset()
        time.sleep(0.05)
