import pickle
from tkinter.filedialog import Open

import cv2
import numpy as np
import gym
from gym import spaces
from random import randrange


class OpenLoopStandard1DTrack(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(OpenLoopStandard1DTrack, self).__init__()    # Define action and observation space
        # They must be gym.spaces objects    # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)    # Example for using image as input:
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(160, 210, 3), dtype=np.uint8
        )
        self.load_data()
        self.last_time = self.data.shape[0]
        self.time = randrange(50)
        self.state = self.data[self.time, :, :, :]
        self.cws = []
        self.alphas = []

    def step(self, action):
        # Execute one time step within the environment
        self.time += 1
        reward = 0
        if action == 1:
            self.cws.append(140.)
            self.alphas.append(1.)
            if self.time >= 335:
                reward = 10
            else:
                reward = -3
        
        done = False
        if self.time == self.last_time - 1:
            done = True
        next_state = self.data[self.time, :, :, :]
        self.state = next_state
        info = {}

        return next_state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        #time, state
        self.time = randrange(50)
        self.state = self.data[self.time, :, :, :]
        self.cws = []
        self.alphas = []

        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ch = 160

        rgb_array = self.state.copy()
        for _ in range(len(self.cws)):
            cw, alpha = self.cws.pop(0), self.alphas.pop(0)
            cw, alpha, rgb_array = self.render_licking(rgb_array, ch, cw, alpha)
            if alpha > 0:
                self.cws.append(cw)
                self.alphas.append(alpha)
            
        cv2.imshow("marked_img", rgb_array)
        cv2.waitKey(1)


    def load_data(self):
        with open('/Users/jhkim/workspace/git/RLbench/src/envs/frames.pkl','rb') as f:
            self.data = pickle.load(f)

    def render_licking(self, img, ch, cw, alpha):
        org = img.copy()
        cv2.line(img, (ch - 25, int(cw)), (ch + 25, int(cw)), (255, 255, 255), 2)
        overlay_img = cv2.addWeighted(img, alpha, org, 1 - alpha, 0)
        cw -= 0.25
        alpha -= 0.01

        return cw, alpha, overlay_img
