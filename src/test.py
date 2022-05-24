import time

import gym
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

from options import get_args


def render(env, model, nstep):
    """ Render how agent interact with environment"""

    obs = env.reset()
    for i in range(nstep): # nstep : 5000
        action, _state = model.predict(obs, deterministic=True) # 0, 1
        obs, reward, done, info = env.step(action)

        plt.figure(3)
        plt.imshow(env.render(mode='rgb_array'))
        plt.title("%s | Step: %d %s" % ("S", i, info))
        plt.axis('off')
        plt.pause(0.08)

        if done:
            obs = env.reset()

    plt.show()

if __name__ == "__main__":
    # trained_model = DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Breakout-v5/a1/a1s1/a1s1r2-42/best_model.zip", verbose=1)
    # env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=0)
    # env = VecFrameStack(env, n_stack=4)
    # render(env, trained_model, 25_000)
    atlantis_agents = [
        PPO.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Atlantis-v5/a1/a1s1/a1s1r1-0/best_model.zip"),
        PPO.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Atlantis-v5/a1/a1s1/a1s1r2-42/best_model.zip"),
        PPO.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Atlantis-v5/a1/a1s1/a1s1r3-53/best_model.zip")
    ]
    krull_agents = [
        PPO.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Krull-v5/a1/a1s1/a1s1r1-0/best_model.zip"),
        PPO.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Krull-v5/a1/a1s1/a1s1r2-42/best_model.zip"),
        PPO.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Krull-v5/a1/a1s1/a1s1r3-53/best_model.zip")
    ]
    kangaroo_agents = [
        PPO.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Kangaroo-v5/a1/a1s1/a1s1r1-0/best_model.zip"),
        PPO.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Kangaroo-v5/a1/a1s1/a1s1r2-42/best_model.zip"),
        PPO.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Kangaroo-v5/a1/a1s1/a1s1r3-53/best_model.zip")
    ]
    # env = make_atari_env(
    #     env_name, n_envs=n_env,
    #     seed=seed, monitor_dir=save_path
    # )
    # env = VecFrameStack(env, n_stack=4) #TODO: set as a hyperparameter
    # env = gym.make("ALE/Atlantis-v5")

    env = make_atari_env("ALE/Kangaroo-v5", n_envs=1)
    env = VecFrameStack(env, n_stack=4)
    render(env, kangaroo_agents[0], 500)