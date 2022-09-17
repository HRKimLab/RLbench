from .oloop1d import OpenLoopStandard1DTrack, OpenLoopPause1DTrack, OpenLoopTeleportLong1DTrack
from .cloop1d import ClosedLoopStandard1DTrack

import gym
import numpy as np


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame (frameskipping)
        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        return self._obs_buffer, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


__all__ = ['oloop1d', 'cloop1d']
