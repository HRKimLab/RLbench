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
import PIL.ImageDraw as ImageDraw
import imageio

BASE_PATH = "../data/"

def get_atari_env(env_name, n_stack=4):
    env = make_atari_env(env_name, n_envs=1)
    env = VecFrameStack(env, n_stack=n_stack)

    return env

def take_snap(env, ax, name, step=0):
    frame = env.render(mode='rgb_array')
    ax.imshow(env.render(mode='rgb_array'))
    ax.text(0.0, 1.01, f"{name} | Step: {step}", transform=ax.transAxes)
    # ax.set_title(f"{name} | Step: {step}")
    ax.axis('off')

    return frame

def snap_finish(ax, name, step):
    ax.text(0.0, 1.01, f"{name} | Step: {step}", transform=ax.transAxes)
    ax.text(
        .5, .5, 'GAME OVER', 
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes
    )
    ax.axis('off')

def mk_fig(q_values):
    bar_plot = plt.bar(list(range(len(q_values))), list(q_values[x] for x in range(len(q_values))))
    plt.xticks(list(range(len(q_values))))
    plt.title('instant q values of actions')
    fig_path = '/home/neurlab-dl1/workspace/RLbench/src/q_value_plot.png'
    plt.savefig(fig_path)
    plt.clf()
    return fig_path

def concat_h(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

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
            if not done[i]: 
                action, _ = model.predict(obs[i], deterministic=True)
                obs[i], _, done[i], info = env.step(action)

                q_values = model.q_net_target(torch.tensor(obs[i]).cuda())[0]
                
                frame = take_snap(env, ax, name, step)
                
                fig_path = mk_fig(q_values)
                plot_figure = Image.open(fig_path)
                frame = concat_h(frame, plot_figure)
                frames.append(frame)


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

    imageio.mimwrite('/home/neurlab-dl1/workspace/RLbench/src/' + env_name + models + names + '.gif', frames, fps=15)

if __name__ == "__main__":
    GAME = "ALE/Breakout-v5"
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
        DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Breakout-v5/a3/a3s1/a3s1r1-0/best_model.zip")
        # PPO.load(p.join(BASE_PATH, GAME, agent_path)) for agent_path in agent_paths
    ]
    # agents = [
    #     PPO.load(p.join(BASE_PATH, GAME, agent_paths[0])),
    #     A2C.load(p.join(BASE_PATH, GAME, agent_paths[1]))
    # ]

    render(GAME, agents, names, 1000)
