""" Single-screen rendering for observing the action decision of single agent """

import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env


def render(env, model, nstep):
    """ Render how agent interact with environment"""

    obs = env.reset()
    for i in range(nstep):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            obs = env.reset()


if __name__ == "__main__":
    trained_model = DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Breakout-v5/a1/a1s1/a1s1r2-42/best_model.zip", verbose=1)
    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    render(env, trained_model, 25_000)
