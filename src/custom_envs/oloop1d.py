import pickle

import cv2
import numpy as np
import gym
from gym import spaces
from random import randrange


class OpenLoop1DTrack(gym.Env):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human']}

    def __init__(self, water_spout, visual_noise=False):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(160, 210, 3), dtype=np.uint8
        )

        self.water_spout = water_spout
        self.visual_noise = visual_noise #TODO

        self.data = self._load_data()
        self.cur_time = randrange(50) # Remove time-bias
        self.start_time = self.cur_time
        self.end_time = self.data.shape[0] - randrange(1, 10) # Black screen
        self.state = self.data[self.cur_time, :, :, :]
        self.licking_cnt = 0

        # For rendering
        self.cws = []
        self.alphas = []

        # For plotting
        self.lick_timing = []
        self.lick_timing_eps = []

    def step(self, action):
        # Execute one time step within the environment
        self.cur_time += 1

        # Next state
        next_state = self.data[self.cur_time, :, :, :]
        # if self.visual_noise: #TODO
        self.state = next_state

        # Reward
        reward = 0
        if action == 1:
            self._licking()
            if (self.cur_time >= self.water_spout) and (self.licking_cnt <= 20):
                reward = 10
            else:
                reward = -5

        # Done
        done = (self.cur_time == self.end_time)

        # Info
        info = {
            "cur_time": self.cur_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "licking_cnt": self.licking_cnt,
            "lick_timing_eps": self.lick_timing_eps
        }

        return next_state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.cur_time = randrange(50)
        self.start_time = self.cur_time
        self.end_time = self.data.shape[0] - randrange(1, 10)
        self.state = self.data[self.cur_time, :, :, :]
        self.licking_cnt = 0

        self.cws = []
        self.alphas = []

        self.lick_timing.append(self.lick_timing_eps)
        self.lick_timing_eps = []

        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ch = 160

        rgb_array = self.state.copy()
        for _ in range(len(self.cws)):
            cw, alpha = self.cws.pop(0), self.alphas.pop(0)
            cw, alpha, rgb_array = self._render_single_lick(rgb_array, ch, cw, alpha)
            if alpha > 0:
                self.cws.append(cw)
                self.alphas.append(alpha)
        
        resized = cv2.resize(rgb_array, (600, 360), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("licking", resized)
        cv2.waitKey(1)

    def _licking(self):
        self.licking_cnt += 1
        self.cws.append(140.)
        self.alphas.append(1.)
        self.lick_timing_eps.append(self.cur_time)

    @staticmethod
    def _render_single_lick(img, ch, cw, alpha):
        org = img.copy()
        cv2.line(img, (ch - 25, int(cw)), (ch + 25, int(cw)), (255, 255, 255), 2)
        overlay_img = cv2.addWeighted(img, alpha, org, 1 - alpha, 0)
        cw -= 0.25
        alpha -= 0.01

        return cw, alpha, overlay_img

    @staticmethod
    def _load_data():
        raise NotImplementedError


class OpenLoopStandard1DTrack(OpenLoop1DTrack):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human']}

    def __init__(self, visual_noise=False):
        super().__init__(
            water_spout=335,
            visual_noise=visual_noise
        )

    @staticmethod
    def _load_data():
        with open(f"custom_envs/track/oloop_standard_1d.pkl", "rb") as f:
            data = pickle.load(f)
        return data


class OpenLoopTeleportLong1DTrack(OpenLoop1DTrack):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human']}

    def __init__(self, visual_noise=False):
        super().__init__(
            water_spout=227,
            visual_noise=visual_noise
        )

    @staticmethod
    def _load_data():
        with open("custom_envs/track/oloop_teleport_long_1d.pkl", "rb") as f:
            data = pickle.load(f)
        return data
