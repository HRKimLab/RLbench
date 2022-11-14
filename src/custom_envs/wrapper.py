import gym
import numpy as np


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip
        self.skip_history = []
        self.skip_history_eps = []

    def step(self, action):
        skip_step = np.random.choice(
            [self._skip - 1, self._skip, self._skip + 1],
            p=[0.2, 0.6, 0.2]
        )

        if action == 1: # Only for oloop
            self.skip_history_eps.append(skip_step)

        total_reward = 0.0
        done = False
        for i in range(skip_step):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer = obs
            total_reward += reward
            if done:
                if action == 1:
                    self.skip_history_eps[-1] = i + 1
                break
        return self._obs_buffer, total_reward, done, info

    def reset(self, **kwargs):
        self.skip_history.append(self.skip_history_eps)
        self.skip_history_eps = []
        return self.env.reset(**kwargs)
