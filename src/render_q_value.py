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

from custom_envs import MaxAndSkipEnv, OpenLoopStandard1DTrack, ClosedLoop1DTrack_virmen
import torch.optim as optim

BASE_PATH = "../data/"

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

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

def get_virmen_mouse_env(n_skip=5):
    env = ClosedLoop1DTrack_virmen()
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

def mk_fig(q_values, y_max, y_min, q_value_history, q_value_history_target, reward_history, nstep, steps, final_steps, td_error, actions, SPOUT, action_list, protocol, SHOCKZONE):
    bar_plot = plt.figure(figsize=(8,8))
    plt.bar(list(range(len(q_values))), list(q_values[x] for x in range(len(q_values))), width = 0.4, color = ['black', 'orange', 'green', 'darkviolet'])
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
    # plt.xticks(list(range(len(q_values))))
    plt.xticks(range(len(action_list)), action_list, fontsize = 25)
    interval = (y_max - y_min) * 0.05
    # plt.ylim(top=y_max + interval, bottom=y_min - interval)
    # plt.ylim([-15, 25])
    # plt.yticks(np.arange(-10,25,10))
    plt.ylim([-2, 2])
    plt.title('Q values of actions', fontsize = 30)
    plt.ylabel('Action Value', fontsize = 25)
    plt.yticks(fontsize = 20)
    bar_plot.tight_layout(pad=0)
    fig_path = 'q_value_bar_plot.png'
    plt.savefig(fig_path)
    plt.close()
    line_plot, ax1 = plt.subplots(figsize=(24,8))
    ax2 = ax1.twinx()
    ax1.set_xlabel('Time Step', fontsize = 30)
    ax1.set_ylabel('Action Value', fontsize = 30)
    colors = ['black', 'orange', 'green', 'darkviolet', 'royalblue']
    colors_target = ['gray', 'wheat', 'yellowgreen', 'thistle', 'skyblue']
    for i in range(len(q_value_history)):
        ax1.plot(list(range(1,steps+1)), q_value_history[i], color = colors[i], label = action_list[i])
        ax1.plot(list(range(1,steps+1)), reward_history[i], color = 'blue')
        # ax1.plot(list(range(1,steps+1)), q_value_history_target[i], color = colors_target[i])
        ax2.plot(list(range(1,steps+1)), td_error, color = 'red', alpha = 0.8, label = 'TD error' if i == 0 else "")
        ax1.legend(fontsize = 25, loc = 'upper right')
        ax2.legend(fontsize = 25, loc = 'lower right')
    plt.xlim([0,nstep])
    # ax1.set_ylim(top = y_max + interval, bottom = y_min - interval)
    # ax1.set_ylim([-15,25])
    # ax2.set_ylim([-15,25])
    ax1.set_ylim([-2,2])
    ax2.set_ylim([-1100,2])
    # ax1.set_yticks(np.arange(-10,25,10))
    # ax2.set_yticks(np.arange(-10,25,10))
    ax1.hlines(y = 0, xmin = 0, xmax = 80, color = 'black', linestyles = '--')
    ax1.tick_params(axis='x', labelsize=25)
    ax1.tick_params(axis = 'y', labelsize = 25)
    ax2.tick_params(axis = 'y', labelsize = 25)
    for spout_timing in SPOUT:
        if protocol == 1:
            ax1.axvline(x=spout_timing, color='blue', linestyle='-', alpha = 0.2, label = 'WATER')
        elif protocol == 2:
            ax1.axvline(x=spout_timing, color='red', linestyle='-', alpha = 0.2, label = 'SHOCK')
    if protocol == 2:
        for shockzone_timing in SHOCKZONE:
            ax1.axvline(x=shockzone_timing, color='green', linestyle='-', label = 'SHOCKZONE')
    fig2_path = 'q_value_line_plot.png'
    plt.savefig(fig2_path)
    plt.close()
    fig, ax3 = plt.subplots(figsize=(24,1))
    for i in range(len(q_value_history)):
        ax3.scatter(list(range(1,steps+1)), [0 for i in range(1,steps+1)], color = [colors[ind] for ind in actions], marker = 's', s=100)
    ax3.axis('off')
    ax3.set_xlim([0,80])
    ax3.set_ylabel('Actions', fontsize = 30)
    # plt.axvline(final_steps, 0,1, linestyle = '--')
    # plt.legend(loc='upper right')
    line_plot.tight_layout(pad=0)
    fig3_path = 'action_plot.png'
    plt.savefig(fig3_path)
    plt.close()
    return fig_path, fig2_path, fig3_path, y_max, y_min

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
    dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height), color = 'white')
    dst.paste(_im1, (-5, 0))
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
        # _im2 = im2.resize(im2.width, int(im1.height/ 2), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((im1.width, int(im2.height * im1.width / im2.width )), resample=resample)
    dst = Image.new('RGB', (_im1.width, _im1.height + _im2.height+10), color = 'white')
    dst.paste(_im1, (-5, 0))
    dst.paste(_im2, (0, _im1.height+10))
    return dst

def render(env_name, model, nstep, action_type, protocol):
    """ Render how agent interact with environment"""
    # env = get_vec_env(env_name)
    # env = env.envs[0]
    # env = get_mouse_env()
    env = get_virmen_mouse_env()
    obs = env.reset()

    done = False
    model = model
    frames = []

    q_value_history = [[] for i in range(env.action_space.n)]
    q_value_history_target = [[] for i in range(env.action_space.n)]
    reward_history = [[] for i in range(env.action_space.n)]
    y_max = np.NaN
    y_min = np.NaN
    steps = 0
    final_steps = []
    td_error = []
    actions = []
    SPOUT = []
    SHOCK = []
    SHOCKZONE = []
    action_list = [["Stop","Lick","Move","Move+Lick"],["Stop","Move"],["Stop", "Lick"],["Stop","Lick","MoveForward","MoveNorthWest","MoveNorthEast","NorthWestLick","NorthEastLick"],
                   ["Stop","JoystickNorth","JoystickNorthEast","JoystickEast","JoystickSouthEast","JoystickSouth","JoystickSouthWest","JoystickWest","JoystickNorthWest"]]

    for _ in tqdm(range(nstep)):
        if done:
            final_steps.append(steps)
            obs = env.reset()
            done = False
        #step
        steps += 1
        frame = env.render(mode='rgb_array')
        action, _ = model.predict(obs, deterministic=True)
        # print(torch.tensor(obs).shape)
        # obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).cuda()
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)
        # obs_tensor = torch.tensor(obs).unsqueeze(0)
        # obs_tensor = torch.tensor(obs)
        q_values = model.q_net(obs_tensor)[0].detach().cpu().tolist()
        # print(q_values)
        # print(action)
        q_values_target = model.q_net_target(obs_tensor)[0].detach().cpu().tolist()
        next_obs, reward , done, info = env.step(action)
        # next_obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).cuda()
        next_obs_tensor = torch.tensor(next_obs).permute(2, 0, 1).unsqueeze(0)
        # next_obs_tensor = torch.tensor(next_obs).unsqueeze(0)
        # next_obs_tensor = torch.tensor(next_obs)
        for i in range(env.action_space.n):
            q_value_history[i].append(q_values[i])
            q_value_history_target[i].append(q_values_target[i])
            reward_history[i].append(reward)
        #td_error
        q_value_predict = model.q_net(obs_tensor)[0].detach().cpu()[action]
        q_value_target = model.q_net_target(next_obs_tensor)[0].detach().cpu().max()
        # reward = how to get S(t+1) reward... step 함수를 써야하는 거 같은데 
        # gamma = 얘도 몇으로..?
        gamma = 0.99
        td_error.append((reward + gamma * q_value_target) - (q_value_predict))
        obs = next_obs
        actions.append(action)
        print(reward)
        if reward > 0:
            SPOUT.append(steps)
        if reward < -50 :
            SHOCK.append(steps)

        try:
            last_step = final_steps[-1]
        except:
            last_step = 0
        if info['shockzone_start_eps'] != []:
            SHOCKZONE.append(last_step + (info['shockzone_start_eps'][0]-1)//5+1)      
        if info['shockzone_end_eps'] != []:
            SHOCKZONE.append(last_step + (info['shockzone_end_eps'][0]-1)//5+1)      


        # make figures and frames
        if protocol == 1:
            fig_path, fig2_path, fig3_path, y_max, y_min = mk_fig(q_values, y_max, y_min, q_value_history, q_value_history_target, reward_history,
                                                    nstep, steps, final_steps, td_error, actions, SPOUT, action_list[action_type], protocol, SHOCKZONE)
        elif protocol == 2:
            fig_path, fig2_path, fig3_path, y_max, y_min = mk_fig(q_values, y_max, y_min, q_value_history, q_value_history_target, reward_history,
                                                    nstep, steps, final_steps, td_error, actions, SHOCK, action_list[action_type], protocol, SHOCKZONE)
        frame = Image.fromarray(frame)
        plot_figure = Image.open(fig_path)
        frame = concat_h_resize(frame, plot_figure)
        plot2_figure = Image.open(fig2_path)
        plot3_figure = Image.open(fig3_path)
        frame = concat_v_resize(frame, plot2_figure)
        frame = concat_v_resize(frame, plot3_figure)
        frames.append(frame)

    imageio.mimwrite('C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\src\\' + str(env_name) + 'dqn' + '.gif', frames, fps=3)
    # imageio.mimwrite('C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\src\\' + str(env_name) + 'dqn' + '.gif', frames, fps=6)


if __name__ == "__main__":
    # GAME = "OpenLoopStandard1DTrack"
    GAME = "ClosedLoop1DTrack_virmen"
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
    # model = [

    #     # DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ClosedLoopStandard1DTrack_P5_N-100/a1/a1s1/a1s1r1-0/best_model.zip")
    #     # DQN.load("/home/neurlab/hyein/RLbench/data/OpenLoopStandard1DTrack/a1/a1s1/a1s1r1-0/best_model")
    #     DQN.load("C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\data\\ClosedLoop1DTrack_virmen\\a2\\a2s1\\a2s1r4-0\\info")
        
    #     # PPO.load(p.join(BASE_PATH, GAME, agent_path)) for agent_path in agent_paths
    # ]
    # model = DQN.load("C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\data\\ClosedLoop1DTrack_virmen\\a2\\a2s1\\a2s1r4-0\info")
    # agents = [
    #     PPO.load(p.join(BASE_PATH, GAME, agent_paths[0])),
    #     A2C.load(p.join(BASE_PATH, GAME, agent_paths[1]))
    # ]

    # model, optimizer, start_epoch = load_ckp("C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\data\\ClosedLoop1DTrack_virmen\\a2\\a2s1\\a2s1r4-0\checkpoint0.pt", model, optim.Adam(model.parameters(), lr=learning_rate))
    
    model = DQN.load("C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\data\\ClosedLoop1DTrack_virmen\\a3\\a3s8\\a3s8r52-42-152456\\info")

    action_type = 1

    protocol = 2 # 1:staircase 2:avoidable shock 3: mixed valence
    
    render(GAME, model, 80, action_type, protocol)