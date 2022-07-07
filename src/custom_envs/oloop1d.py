import pickle

import cv2
import numpy as np
import gym
import imageio
from gym import spaces
from random import randrange


class OpenLoop1DTrack(gym.Env):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human', 'gif']}

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
        self.bar_states = []
        self.frames = []

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

        # Water Spout rendering
        if self.cur_time == self.water_spout:
            self.bar_states.append((140., 1., True))

        return next_state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.cur_time = randrange(50)
        self.start_time = self.cur_time
        self.end_time = self.data.shape[0] - randrange(1, 10)
        self.state = self.data[self.cur_time, :, :, :]
        self.licking_cnt = 0

        self.bar_states = []

        self.lick_timing.append(self.lick_timing_eps)
        self.lick_timing_eps = []

        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ch = 160

        rgb_array = self.state.copy()
        for _ in range(len(self.bar_states)):
            cw, alpha, is_spout = self.bar_states.pop(0)
            cw, alpha, rgb_array = self._render_single_bar(
                rgb_array, ch, cw, alpha, 
                color='red' if is_spout else 'white'
            )
            if alpha > 0:
                self.bar_states.append((cw, alpha, is_spout))
        
        resized = cv2.resize(rgb_array, (600, 360), interpolation=cv2.INTER_CUBIC)
        if mode == 'human':
            cv2.imshow("licking", resized)
            cv2.waitKey(1)
        elif mode == 'gif':
            self.frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    def save_gif(self):
        imageio.mimsave('video.gif', self.frames, fps=60)

    def _licking(self):
        self.licking_cnt += 1
        self.bar_states.append((140., 1., False))
        self.lick_timing_eps.append(self.cur_time)

    @staticmethod
    def _render_single_bar(img, ch, cw, alpha, color):
        org = img.copy()
        cv2.line(
            img, (ch - 25, int(cw)), (ch + 25, int(cw)), 
            (255, 255, 255) if color == 'white' else (0, 0, 255), 1
        )
        overlay_img = cv2.addWeighted(img, alpha, org, 1 - alpha, 0)
        cw -= 0.7
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
