""" Multi-screen rendering & saving for comparing multiple agents """
import os.path as p
from celluloid import Camera
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from tqdm import tqdm
from PIL import Image
import PIL.ImageDraw as ImageDraw
import imageio
BASE_PATH = "../data/"
def get_atari_env(env_name, n_stack=4):
    env = make_atari_env(env_name, n_envs=1)
    env = VecFrameStack(env, n_stack=n_stack)
    return env
def get_vec_env(env_name, n_stack=4):
    env = make_vec_env(env_name, n_envs = 1)
    env = VecFrameStack(env, n_stack= n_stack)
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
def mk_fig(q_values, y_max, q_value_history, nstep, steps):
    bar_plot = plt.bar(list(range(len(q_values))), list(q_values[x] for x in range(len(q_values))))
    if np.isnan(y_max):
            y_max = max(q_values)
    else:
        if max(q_values) > y_max:
            y_max = max(q_values)
    plt.xticks(list(range(len(q_values))))
    plt.ylim(top=y_max)
    plt.title('instant q values of actions')
    fig_path = '/home/neurlab-dl1/workspace/RLbench/src/q_value_bar_plot.png'
    plt.savefig(fig_path)
    plt.clf()
    line_plot = plt.figure()
    for i in range(len(q_value_history)):
        plt.plot(list(range(1,steps+1)), q_value_history[i])
    plt.xlim([0,nstep])
    plt.ylim(top = y_max)
    fig2_path = '/home/neurlab-dl1/hyein/python_study_notes/Hyeeiin/reinforcement_learning/DQN_pytorch/q_value_line_plot.png'
    plt.savefig(fig2_path)
    plt.clf()
    return fig_path, fig2_path, y_max
def concat_h_resize(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
    if im1.height == im2.height:
        _im1 = im1
        _im2 = im2
    elif (((im1.height > im2.height) and resize_big_image) or
          ((im1.height < im2.height) and not resize_big_image)):
        _im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
    dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (_im1.width, 0))
    return dst
def concat_v_resize(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
    if im1.width == im2.width:
        _im1 = im1
        _im2 = im2.resize(im2.width, int(im1.height/ 2), resample = resample)
    elif (((im1.width > im2.width) and resize_big_image) or
          ((im1.width < im2.width) and not resize_big_image)):
        _im1 = im1.resize((im2.width, int(im1.height * im2.width / im1.width)), resample=resample)
        _im2 = im2.resize(im2.width, int(im1.height/2), resample = resample)
    else:
        _im1 = im1
        _im2 = im2.resize((im1.width, int(im2.height * im1.width / im2.width / 2)), resample=resample)
    dst = Image.new('RGB', (_im1.width, _im1.height + _im2.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (0, _im1.height))
    return dst
def render(env_name, model, name, nstep):
    """ Render how agent interact with environment"""
    # fig_num = len(models)
    # fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    # plt.subplots_adjust(wspace=0.5)
    # camera = Camera(fig)
    env = get_vec_env(env_name)
    env = env.envs[0]
    obs = env.reset()
    # done = [False] * fig_num
    # final_steps = [0] * fig_num
    done = False
    final_steps = [0]
    model = model[0]
    frames = []
    q_value_history_0 =[]
    q_value_history_1 = []
    q_value_history = [[] for i in range(env.action_space.n)]
    y_max = np.NaN
    steps = 0
    for step in tqdm(range(nstep)):
        if not done:
            steps += 1
            frame = env.render(mode='rgb_array')
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            observation = obs.reshape((-1,) + model.observation_space.shape)
            observation = torch.tensor(observation).cuda()
            # get q_values
            # q_values = tensor([[0.1247, -0.0102]])
            q_values = model.q_net_target(observation)[0].detach().cpu().tolist()
            for i in range(env.action_space.n):
                q_value_history[i].append(model.q_net_target(observation)[0].detach().cpu().numpy()[i])
            # q_value_history_0.append(model.q_net_target(observation)[0].detach().cpu().numpy()[0])
            # q_value_history_1.append(model.q_net_target(observation)[0].detach().cpu().numpy()[1])
            #make figures and frames
            fig_path, fig2_path, y_max = mk_fig(q_values, y_max, q_value_history, nstep, steps)
            frame = Image.fromarray(frame)
            plot_figure = Image.open(fig_path)
            frame = concat_h_resize(frame, plot_figure)
            plot2_figure = Image.open(fig2_path)
            frame = concat_v_resize(frame, plot2_figure)
            frames.append(frame)
        else:
            # final_steps = step
            obs = env.reset()
            done = False
    imageio.mimwrite('/home/neurlab-dl1/workspace/RLbench/src/' + str(env_name) + str(model) + '.gif', frames, fps=15)
if __name__ == "__main__":
    # GAME = "ALE/Breakout-v5"
    GAME = "CartPole-v1"
    agent_paths = [
        # "a1/a1s1/a1s1r1-0/rl_model_50000_steps",
        # "a1/a1s1/a1s1r1-0/rl_model_100000_steps",
        # "a1/a1s1/a1s1r1-0/rl_model_150000_steps",
        # "a1/a1s1/a1s1r1-0/rl_model_300000_steps",
        # "a1/a1s1/a1s1r2-42/rl_model_500000_steps",
        # "a1/a1s1/a1s1r2-42/rl_model_700000_steps",
        # "a1/a1s1/a1s1r4-7/rl_model_800000_steps",
        # "a1/a1s1/a1s1r4-7/best_model",
        "a1/a1s1/a1s1r1-0/best_model"
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
    model = [
        # DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Breakout-v5/a3/a3s1/a3s1r1-0/best_model.zip")
        DQN.load("/home/neurlab-dl1/workspace/RLbench/data/CartPole-v1/a1/a1s1/a1s1r1-0/best_model.zip")
        # PPO.load(p.join(BASE_PATH, GAME, agent_path)) for agent_path in agent_paths
    ]
    # agents = [
    #     PPO.load(p.join(BASE_PATH, GAME, agent_paths[0])),
    #     A2C.load(p.join(BASE_PATH, GAME, agent_paths[1]))
    # ]
    render(GAME, model, names, 1000)