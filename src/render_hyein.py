""" Multi-screen rendering & saving for comparing multiple agents """
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
import imageio
from PIL import Image
from tqdm import tqdm

from custom_envs import OpenLoopStandard1DTrack, MaxAndSkipEnv

BASE_PATH = "../data/"

def get_atari_env(env_name, n_stack=4):
    env = make_atari_env(env_name, n_envs=1)
    env = VecFrameStack(env, n_stack=n_stack)
    return env

def get_vec_env(env_name, n_stack=4):
    env = make_vec_env(env_name, n_envs = 1)
    env = VecFrameStack(env, n_stack= n_stack)
    return env

def get_mouse_env(n_skip=5):
    env = OpenLoopStandard1DTrack()
    env = MaxAndSkipEnv(env, skip=n_skip)
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

def mk_fig(q_values, y_max, y_min, q_value_history, nstep, steps, final_steps, td_error):
    bar_plot = plt.figure(figsize=(6,8))
    plt.bar(list(range(len(q_values))), list(q_values[x] for x in range(len(q_values))), width = 0.25)
    if np.isnan(y_max):
            y_max = max(q_values)
    else:
        if max(q_values) > y_max:
            y_max = max(q_values)
    if np.isnan(y_min):
            y_min = min(q_values)
    else:
        if min(q_values) < y_min:
            y_min = min(q_values)
    plt.xticks(list(range(len(q_values))))
    interval = (y_max - y_min) * 0.05
    plt.ylim(top=y_max + interval, bottom=y_min - interval)
    plt.title('instant q values of actions')
    bar_plot.tight_layout(pad=0)
    fig_path = 'q_value_bar_plot.png'
    plt.savefig(fig_path)
    plt.close()
    line_plot, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    colors = ['black', 'orange', 'green', 'darkviolet', 'skyblue']
    for i in range(len(q_value_history)):
        ax1.plot(list(range(1,steps+1)), q_value_history[i], color = colors[i])
    # ax2.plot(list(range(1,steps+1)), td_error, color = 'orange', alpha = 0.5)
    for i in range(1, steps +1):
        if steps <= 10:
            ax2.plot(i, sum(td_error[:i+1])/i)
        else:
            ax2.plot(i, sum(td_error[i-10:i+1])/10)

    plt.xlim([0,nstep])
    ax1.set_ylim(top = y_max + interval, bottom = y_min - interval)
    # plt.axvline(final_steps, 0,1, linestyle = '--')
    # plt.legend(loc='upper right')
    line_plot.tight_layout(pad=0)
    fig2_path = 'q_value_line_plot.png'
    plt.savefig(fig2_path)
    plt.close()
    return fig_path, fig2_path, y_max, y_min

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

# def concat_v_resize(im1, im2, resample=Image.BICUBIC, resize_big_image=True):
#     if im1.width == im2.width:
#         _im1 = im1
#         _im2 = im2.resize(im2.width, int(im1.height/ 2), resample=resample)
#     elif (((im1.width > im2.width) and resize_big_image) or
#           ((im1.width < im2.width) and not resize_big_image)):
#         _im1 = im1.resize((im2.width, int(im1.height * im2.width / im1.width)), resample=resample)
#         _im2 = im2.resize((im2.width, int(im1.height/2)), resample=resample)
#     else:
#         _im1 = im1
#         _im2 = im2.resize((im1.width, int(im2.height * im1.width / im2.width / 2)), resample=resample)
#     dst = Image.new('RGB', (_im1.width, _im1.height + _im2.height))
#     dst.paste(_im1, (0, 0))
#     dst.paste(_im2, (0, _im1.height))
#     return dst

def concat_v_resize(im1, im2, resample=Image.BICUBIC):
    if im1.width == im2.width:
        _im1 = im1
        _im2 = im2.resize(im2.width, int(im1.height/ 2), resample=resample)
    else:
        _im1 = im1
        _im2 = im2.resize((im1.width, int(im2.height * im1.width / im2.width / 2)), resample=resample)
    dst = Image.new('RGB', (_im1.width, _im1.height + _im2.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (0, _im1.height))
    return dst

def render(env_name, model, nstep):
    """ Render how agent interact with environment"""
    # env = get_vec_env(env_name)
    # env = env.envs[0]
    env = get_mouse_env()
    obs = env.reset()

    done = False
    model = model[0]
    frames = []

    q_value_history = [[] for i in range(env.action_space.n)]
    y_max = np.NaN
    y_min = np.NaN
    steps = 0
    final_steps = []
    td_error = []

    for _ in tqdm(range(nstep)):
        if done:
            final_steps.append(steps)
            obs = env.reset()
            done = False
        #step
        steps += 1
        frame = env.render(mode='rgb_array')
        action, _ = model.predict(obs, deterministic=True)
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).cuda()
        q_values = model.q_net(obs_tensor)[0].detach().cpu().tolist()
        next_obs, reward , done, _ = env.step(action)
        next_obs_tensor = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0).cuda()
        for i in range(env.action_space.n):
            q_value_history[i].append(q_values[i])
        #td_error
        q_value_predict = model.q_net(obs_tensor)[0].detach().cpu()[action]
        q_value_target = model.q_net_target(next_obs_tensor)[0].detach().cpu().max()
        # reward = how to get S(t+1) reward... step 함수를 써야하는 거 같은데 
        # gamma = 얘도 몇으로..?
        gamma = 0.9
        td_error.append((reward + gamma * q_value_target) - (q_value_predict))
        obs = next_obs

        # make figures and frames
        fig_path, fig2_path, y_max, y_min = mk_fig(q_values, y_max, y_min, q_value_history, nstep, steps, final_steps, td_error)
        frame = Image.fromarray(frame)
        plot_figure = Image.open(fig_path)
        frame = concat_h_resize(frame, plot_figure)
        plot2_figure = Image.open(fig2_path)
        frame = concat_v_resize(frame, plot2_figure)
        frames.append(frame)

    imageio.mimwrite('/home/neurlab-dl1/workspace/RLbench/src/' + str(env_name) + str(model) + '.gif', frames, fps=15)


if __name__ == "__main__":
    GAME = "OpenLoopStandard1DTrack"
    agent_paths = [
        # "a1/a1s1/a1s1r1-0/rl_model_50000_steps",
        # "a1/a1s1/a1s1r1-0/rl_model_100000_steps",
        # "a1/a1s1/a1s1r1-0/rl_model_150000_steps",
        # "a1/a1s1/a1s1r1-0/rl_model_300000_steps",
        # "a1/a1s1/a1s1r2-42/rl_model_500000_steps",
        # "a1/a1s1/a1s1r2-42/rl_model_700000_steps",
        # "a1/a1s1/a1s1r4-7/rl_model_800000_steps",
        # "a1/a1s1/a1s1r4-7/best_model",
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

        DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ClosedLoopStandard1DTrack_P10_N-5/a1/a1s1/a1s1r1-0/best_model.zip")
        # DQN.load("/home/neurlab/hyein/RLbench/data/OpenLoopStandard1DTrack/a1/a1s1/a1s1r1-0/best_model")
        # DQN.load("/home/neurlab-dl1/workspace/RLbench/data/OpenLoopStandard1DTrack/a1/a1s1/a1s1r1-0/best_model")
        
        # PPO.load(p.join(BASE_PATH, GAME, agent_path)) for agent_path in agent_paths
    ]
    # agents = [
    #     PPO.load(p.join(BASE_PATH, GAME, agent_paths[0])),
    #     A2C.load(p.join(BASE_PATH, GAME, agent_paths[1]))
    # ]
    render(GAME, model, 1000)