import pickle

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

    def step(self, action):
        # Execute one time step within the environment
        self.time += 1
        reward = 0
        if action == 1:
            if self.time >= 335:
                reward = 5
            else:
                reward = -3
        
        done = False
        if self.time == self.last_time:
            done = True
        next_state = self.data[self.time, :, :, :]
        self.state = next_state
        return next_state, reward, done, 0

    def reset(self):
        # Reset the state of the environment to an initial state
        #time, state
        self.time = randrange(50)
        self.state = self.data[self.time, :, :, :]

        return self.state, self.time

    #def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...

    def load_data(self):
        with open('frames.pkl','rb') as f:
            self.data = pickle.load(f)

env = OpenLoopStandard1DTrack()