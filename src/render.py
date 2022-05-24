""" Multi-screen rendering & saving for comparing multiple agents """

import os
import os.path as p
from celluloid import Camera

import gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

BASE_PATH = "/Users/jhkim/Downloads/ar"

def get_atari_env(env_name, n_stack=4):
    env = make_atari_env(env_name, n_envs=1)
    env = VecFrameStack(env, n_stack=n_stack)

    return env

def take_snap(env, ax, name, step=0):
    ax.imshow(env.render(mode='rgb_array'))
    ax.text(0.0, 1.01, f"{name} | Step: {step}", transform=ax.transAxes)
    # ax.set_title(f"{name} | Step: {step}")
    ax.axis('off')

def snap_finish(ax, name, step):
    ax.text(0.0, 1.01, f"{name} | Step: {step}", transform=ax.transAxes)
    ax.text(
        .5, .5, 'GAME OVER', 
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes
    )
    ax.axis('off')

def render(env_name, models, names, nstep):
    """ Render how agent interact with environment"""

    fig_num = len(models)
    fig, axs = plt.subplots(1, fig_num, figsize=(15, 5))
    plt.subplots_adjust(wspace=1)
    camera = Camera(fig)

    envs = [get_atari_env(env_name) for _ in range(len(models))]
    obs = [env.reset() for env in envs]
    done = [False] * fig_num
    final_steps = [0] * 3

    delay = 0
    for step in range(nstep):
        for i, (ax, env, name, model) in enumerate(zip(axs, envs, names, models)):
            if not done[i]: 
                action, _ = model.predict(obs[i], deterministic=True)
                obs[i], _, done[i], info = env.step(action)
                take_snap(env, ax, name, step)
                if done:
                    final_steps[i] = step
            else:
                snap_finish(ax, name, final_steps[i])
        if all(done):
            delay += 1

        camera.snap()
        if delay == 10:
            break

    animation = camera.animate()
    animation.save("animation.mp4", fps=10)

if __name__ == "__main__":
    atlantis_agents = [
        PPO.load(p.join(BASE_PATH, "Atlantis-v5/a1/a1s1/a1s1r1-0/best_model.zip")),
        PPO.load(p.join(BASE_PATH, "Atlantis-v5/a1/a1s1/a1s1r2-42/best_model.zip")),
        PPO.load(p.join(BASE_PATH, "Atlantis-v5/a1/a1s1/a1s1r3-53/best_model.zip"))
    ]

    breakout_agents = [
        PPO.load(p.join(BASE_PATH, "Breakout-v5/a1/a1s1/a1s1r1-0/best_model.zip")),
        PPO.load(p.join(BASE_PATH, "Breakout-v5/a1/a1s1/a1s1r2-42/best_model.zip")),
        PPO.load(p.join(BASE_PATH, "Breakout-v5/a1/a1s1/a1s1r3-53/best_model.zip"))
    ]
    kangaroo_agents = [
        PPO.load(p.join(BASE_PATH, "Kangaroo-v5/a1/a1s1/a1s1r1-0/best_model.zip")),
        PPO.load(p.join(BASE_PATH, "Kangaroo-v5/a1/a1s1/a1s1r2-42/best_model.zip")),
        PPO.load(p.join(BASE_PATH, "Kangaroo-v5/a1/a1s1/a1s1r3-53/best_model.zip"))
    ]

    names = ["a1s1r1-0", "a1s1r2-42", "a1s1r3-53"]

    render("ALE/Atlantis-v5", atlantis_agents, names, 500)
