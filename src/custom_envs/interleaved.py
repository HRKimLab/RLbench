import random

import gym
from gym import spaces
import numpy as np

from .wrapper import MaxAndSkipEnv
from .oloop1d import (
    OpenLoopStandard1DTrack,
    OpenLoopPause1DTrack,
    OpenLoopTeleportLong1DTrack
)


class InterleavedOpenLoop1DTrack(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, n_env=3, pos_rew=10, neg_rew=-5):
        """
            n_env:
                2 - Standard, Pause
                3 - Standard, Pause, Long Teleport #TODO
        """
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(160, 210, 3), dtype=np.uint8
        )
        self.env_set = [
            OpenLoopStandard1DTrack(pos_rew=pos_rew, neg_rew=neg_rew),
            OpenLoopPause1DTrack(pos_rew=pos_rew, neg_rew=neg_rew),
            OpenLoopTeleportLong1DTrack(pos_rew=pos_rew, neg_rew=neg_rew)
        ][:n_env]
        self.env_set = [MaxAndSkipEnv(env, skip=5) for env in self.env_set]
        self.n_env = n_env

        self.cur_n = None
        self.cur_env = None
        self.env_history = []
        self.env_prog_time = []
        
    def step(self, action):
        return self.cur_env.step(action)

    def reset(self, stochasticity=None):
        return self._reset(stochasticity)

    def render(self, mode='human'):
        return self.cur_env.render(mode)

    def _reset(self, stochasticity=None):
        self.cur_n = random.randint(0, self.n_env - 1)
        self.cur_env = self.env_set[self.cur_n]
        self.env_history.append(self.cur_n)

        next_state = self.cur_env.reset(stochasticity=stochasticity)
        self.env_prog_time.append(self.cur_env.end_time - self.cur_env.start_time)

        return next_state
