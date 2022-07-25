""" Multi-screen rendering & saving for comparing multiple agents """

import os.path as p
from celluloid import Camera
import torch

import matplotlib.pyplot as plt
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from tqdm import tqdm
from PIL import Image
from ale_py import ALEInterface

ale = ALEInterface()


BASE_PATH = "../data/"

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
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.5)
    camera = Camera(fig)

    envs = [get_atari_env(env_name) for _ in range(len(models))]
    obs = [env.reset() for env in envs]
    done = [False] * fig_num
    final_steps = [0] * fig_num

    delay = 0
    frames = []
    for step in tqdm(range(nstep)):
        for i, (env, name, model) in enumerate(zip(envs, names, models)):
            ax = axs[i // 4][i % 4]
            print(model.q_net_target)
            if not done[i]: 
                action, _ = model.predict(obs[i], deterministic=True)
                obs[i], _, done[i], info = env.step(action)
                
                #q_values = model.q_net_target(torch.tensor(obs[i]).cuda())[0]
                
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
    GAME = "ALE/Breakout-v0"
    agent_paths = [
        # "a1/a1s1/a1s1r1-0/rl_model_50000_steps",
        # "a1/a1s1/a1s1r1-0/rl_model_100000_steps",
        # "a1/a1s1/a1s1r1-0/rl_model_150000_steps",
        # "a1/a1s1/a1s1r1-0/rl_model_300000_steps",
        # "a1/a1s1/a1s1r2-42/rl_model_500000_steps",
        # "a1/a1s1/a1s1r2-42/rl_model_700000_steps",
        # "a1/a1s1/a1s1r4-7/rl_model_800000_steps",
        # "a1/a1s1/a1s1r4-7/best_model",
        "a1/a1s1/a1s1r3-53/best_model"
    ]

    # agent_paths = ["a1/a1s1/a1s1r1-0/best_model", "a2/a2s1/a2s1r1-0/best_model"]
    names = [
        # "0.05M steps",
        "0.1M steps"
        # "0.15M steps",
        # "0.2M steps",
        # "0.4M steps",
        # "0.6M steps",
        # "0.8M steps",
        # "1.0M steps",
    ]

    agents = [ 
        DQN.load("/home/neurlab/yw/RLbench/data/ALE/Breakout-v5/a3/a3s1/a3s1r1-0/best_model.zip")
        # PPO.load(p.join(BASE_PATH, GAME, agent_path)) for agent_path in agent_paths
    ]
    # agents = [
    #     PPO.load(p.join(BASE_PATH, GAME, agent_paths[0])),
    #     A2C.load(p.join(BASE_PATH, GAME, agent_paths[1]))
    # ]

    render(GAME, agents, names, 1000)
