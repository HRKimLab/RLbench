import time
import copy

import gym
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env


def get_atari_env(env_name, n_stack=4):
    env = make_atari_env(env_name, n_envs=1)
    env = VecFrameStack(env, n_stack=n_stack)

    return env

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (env._spec.id,step, info))
    plt.axis('off')


def show_state(env, nrow, ncol, nidx, step=0, info=""):
    plt.subplot(nrow, ncol, nidx)
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % ("S", step, info))
    plt.axis('off')
    plt.pause(0.08)


def render(env_name, models, nstep):
    """ Render how agent interact with environment"""

    envs = [get_atari_env(env_name) for _ in range(len(models))]
    obs = [env.reset() for env in envs]
    done = [False] * len(models)
    for _ in range(nstep): # nstep : 5000
        for i, (env, model) in enumerate(zip(envs, models)):
            action, _ = model.predict(obs[i], deterministic=True)
            obs[i], _, done[i], info = env.step(action)
            show_state(env, len(models), 1, i+1)
            
            if done:
                obs[i] = env.reset()


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

    render("ALE/Kangaroo-v5", kangaroo_agents, 500)
