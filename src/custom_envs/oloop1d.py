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

    def __init__(self, water_spout, video_path, visual_noise=False, pos_rew=10, neg_rew=-5):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(160, 210, 3), dtype=np.uint8
        )

        self.water_spout = water_spout
        self.video_path = video_path
        self.visual_noise = visual_noise #TODO

        self.data = self._load_data()
        self.cur_time = randrange(50) # Remove time-bias
        self.start_time = self.cur_time
        self.end_time = self.data.shape[0] - randrange(1, 10) # Black screen
        self.state = self.data[self.cur_time, :, :, :]
        self.licking_cnt = 0

        # For rendering
        self.frames = []
        self.original_frames = self._get_original_video_frames() # (682, 1288)
        self.mice_pic = self._load_mice_image()

        # For plotting
        self.lick_timing = []
        self.lick_timing_eps = []

        # Temporal variables (for experiments)
        self.pos_rew = pos_rew
        self.neg_rew = neg_rew
        self.reward_set = []
        self.reward_set_eps = []

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
                reward = self.pos_rew
            else:
                reward = self.neg_rew
        self.reward_set_eps.append(reward)
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

        self.reward_set.append(self.reward_set_eps)
        self.reward_set_eps = []

        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        rgb_array = self.original_frames[self.cur_time, :, :, :].copy()
        height, width, _ = rgb_array.shape

        pos_template = (width - 40) / (self.end_time - self.start_time)

        # Upper padding
        padding_height = height // 8
        rgb_array = cv2.copyMakeBorder(rgb_array, padding_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Base line
        base_height = padding_height // 2
        cv2.line(rgb_array, (20, base_height), (width - 20, base_height), (0, 0, 0), 2)
        
        # Current position
        x_offset = int((self.cur_time - self.start_time) * pos_template)
        y_offset = base_height - 20
        rgb_array[y_offset:y_offset+self.mice_pic.shape[0], x_offset:x_offset+self.mice_pic.shape[1], :] = self.mice_pic

        # Licking
        for lick_timing in self.lick_timing_eps:
            lick_x_pos = 20 + int((lick_timing - self.start_time) * pos_template)
            cv2.line(rgb_array, (lick_x_pos, base_height - 30), (lick_x_pos, base_height + 30), (0, 0, 0), 1)

        # Water spout
        spout_x_pos = 20 + int((self.water_spout - self.start_time) * pos_template)
        cv2.line(rgb_array, (spout_x_pos, base_height - 30), (spout_x_pos, base_height + 30), (255, 0, 0), 3)

        if mode == 'human':
            cv2.imshow("licking", rgb_array)
            cv2.waitKey(1)
        elif mode == 'gif':
            self.frames.append(cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB))
        elif mode == 'mp4':
            self.frames.append(rgb_array)
        elif mode == 'rgb_array':
            return cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)

    def save_gif(self):
        imageio.mimsave('video.gif', self.frames, duration=0.005)

    def save_mp4(self, name="test.mp4"):
        height, width, _ = self.frames[0].shape
        fps = 60

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(name, fourcc, float(fps), (width, height))
        for frame in self.frames:
            video.write(frame)
        video.release()

    def _licking(self):
        self.licking_cnt += 1
        self.lick_timing_eps.append(self.cur_time)

    def _get_original_video_frames(self):
        capture = cv2.VideoCapture(self.video_path)

        frames = []
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frames.append(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        frames = np.stack(frames, axis=0)

        return frames

    @staticmethod
    def _load_mice_image():
        mice_pic = cv2.imread("../assets/mice.png", cv2.IMREAD_UNCHANGED)
        mice_pic = cv2.resize(mice_pic, (40, 40))
        mice_pic = np.repeat((mice_pic[:, :, 3] < 50).reshape(40, 40, 1), repeats=3, axis=2) * 255
        return mice_pic

    @staticmethod
    def _load_data():
        raise NotImplementedError


class OpenLoopStandard1DTrack(OpenLoop1DTrack):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human']}

    def __init__(self, visual_noise=False, pos_rew=10, neg_rew=-5):
        super().__init__(
            water_spout=335,
            video_path="custom_envs/track/VR_standard.mp4",
            visual_noise=visual_noise,
            pos_rew=pos_rew,
            neg_rew=neg_rew
        )

    @staticmethod
    def _load_data():
        with open(f"custom_envs/track/oloop_standard_1d.pkl", "rb") as f:
            data = pickle.load(f)
        return data


class OpenLoopTeleportLong1DTrack(OpenLoop1DTrack):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human']}

    def __init__(self, visual_noise=False, pos_rew=10, neg_rew=-5):
        super().__init__(
            water_spout=227,
            video_path="custom_envs/track/VR_tele_1dest_long.mp4",
            visual_noise=visual_noise,
            pos_rew=pos_rew,
            neg_rew=neg_rew
        )

    @staticmethod
    def _load_data():
        with open("custom_envs/track/oloop_teleport_long_1d.pkl", "rb") as f:
            data = pickle.load(f)
        return data
