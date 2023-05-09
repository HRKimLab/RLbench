import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from utils import get_algo_from_agent
from options import get_args_licking
from utils.options import get_args

FPS = 39.15
LICK_PER_SEC = 8
WINDOW_SIZE = 5
SPOUT = 80

def plot_licking(args):
    """ Plot the behavior (licking) data of mouse agent """

    data_path = Path(os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data')))
    data_path = data_path / args.env

    try:
        for _ in range(2):
            dir_list = os.listdir(data_path)
            for dir_name in dir_list:
                if dir_name in args.agent:
                    data_path /= dir_name
                    break

        dir_list = os.listdir(data_path)
        for dir_name in dir_list:
            if args.agent in dir_name:
                data_path /= dir_name
                break
    except:
        raise FileNotFoundError("Given agent name is not found.")
    
    spout_path = data_path / "spout_timing.pkl"
    licking_path = data_path / "lick_timing.pkl"
    actions_path = data_path / "actions.pkl"
    trial_start_path = data_path / "trial_start_pos.pkl"

    date_today = date.today().isoformat()
    # track_len = 100

    spout_data = pd.read_pickle(spout_path)
    # print(spout_data)
    SPOUT = int((max(sum(spout_data,[]))-1)//5 +1)
    # print(SPOUT)

    licking_data = pd.read_pickle(licking_path)
    # print(licking_data)

    actions_data = pd.read_pickle(actions_path)
    # print(actions_data)
    # lengths = [len(actions_data[i]) for i in range(len(actions_data))]
    # # print(lengths)
    # length = int(max(lengths))
    # # print(length)

    trial_start_data = pd.read_pickle(trial_start_path)
    print(trial_start_data)

    lengths = []
    for i in range(len(actions_data)):
        # print(spout_data[i])
        try:
            lengths.append(int(len(actions_data[i])//5 - spout_data[i][0]//5))
        except:
            continue
    length = SPOUT + max(lengths) + 2
    # print(length)

    # data = np.zeros((len(licking_data), track_len))
    data = np.zeros((len(licking_data), length))
    dis_list = {}
    dis_list2 = {}
    x_nor = []
    count = 0

    # Dotplot - licking
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    for i, l in enumerate(licking_data):

        if len(spout_data[i]) == 0:
            continue
        spout_nor2 = (spout_data[i][0] -1)//5 +1
        dis2 = spout_nor2 - SPOUT
        dis_list2[i] = dis2
        
        # if (len(spout_data[i]) == 0 or len(l)==0):
        #     continue

        if len(l) == 0:
            continue
        
        x = l[::5]
        x = [(ele-1)/5+1 for ele in x]

        spout_nor = (spout_data[i][0] -1)//5 +1
        dis = spout_nor - SPOUT
        # print(dis)
        dis_list[i] = dis

        x = [int(ele-dis) for ele in x]
        # print(x)
        # xn = np.array(x)
        # print(xn)
        # upper_index = np.where(xn<=100)[0][-1]
        # # print(np.where(xn<=100)[0][-1])
        # lower_index = np.where(xn>=0)[0][0]
        # x = x[lower_index:upper_index]
        y = [i] * len(x)

        ax1.scatter(x, y, color='black', s=1)
        data[i, x] = 1
    ax1.set_title("Lick")
    ax1.axvline(x=SPOUT, color='r', linestyle='-')
    ax1.get_xaxis().set_visible(False)
    # ax1.set_xlim(0,100)
    ax1.invert_yaxis()

    # Lineplot
    line_y = []
    # for i in range(track_len - LICK_PER_SEC):
    for i in range(length - LICK_PER_SEC):
        half_window = LICK_PER_SEC // 2
        line_y.append(data[:, i-half_window:i+half_window].sum() // LICK_PER_SEC)
    ax2.plot(line_y)
    ax2.set_xlabel("Steps", fontsize=10)
    ax2.set_ylabel("Lick (licks/s)", fontsize=10)
    ax2.axvline(x=SPOUT, color='r', linestyle='-')

    algo_name, _ = get_algo_from_agent(args.agent, data_path)
    fig.suptitle(f"{args.agent} ({algo_name.upper()}) / {args.env} / {date_today}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    COLORS = ["black", "orange", "green", "darkviolet"]
    actions_nor = []

    #Dotplot2 - action-step
    fig2, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    for i,l in enumerate(actions_data):
        if (len(spout_data[i]) == 0 or len(l)==0):
            continue

        x = range(1,length)
        actions = l[::5]

        #shift x or actions to match SPOUT and the spout_data
        if (i not in dis_list2.keys()):
            continue
        dis = -dis_list2[i]
        x = x[dis:]
        x = x[:len(actions)]
        x_nor.append(x)
        # print(actions.count(1))
        actions_nor.append(actions)
        y = [i] * len(x)

        ax1.scatter(x,y,color = [COLORS[ind] for ind in actions],s=1)
    ax1.set_title("Actions(orange:lick, green:move, purple:move+lick, black: no action)")
    ax1.axvline(x=SPOUT, color='r', linestyle='-')
    ax1.get_xaxis().set_visible(False)
    # ax1.set_xlim(0,100)
    ax1.invert_yaxis()   

    #Lineplot2 - action-step
    # line_y2 = [[0 for i in range(100)] for j in range(3)]
    line_y2 = [[0 for i in range(length)] for j in range(4)]
    all_actions_num = [0 for i in range(length)]
    for i in range(len(x_nor)):
        for j in range(len(x_nor[i])):
            line_y2[actions_nor[i][j]][x_nor[i][j]] += 1
            all_actions_num[x_nor[i][j]] += 1
    for i in range(4):
        for j in range(length):
            if all_actions_num[j] == 0:
                continue
            line_y2[i][j] /= all_actions_num[j]
    # print(actions_nor)
    for i in range(4):
        ax2.plot(line_y2[i], color = COLORS[i])
    ax2.set_xlabel("Steps", fontsize=10)
    ax2.set_ylabel("Actions(averaged)", fontsize=10)
    ax2.axvline(x=SPOUT, color='r', linestyle='-')

    fig2.suptitle(f"{args.agent} ({algo_name.upper()}) / {args.env} / {date_today}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    x_2 = []
    actions_2 = []
    ac_lengths = [len(actions_data[i])//5 for i in range(len(actions_data))]
    ac_length = max(ac_lengths) + 2
    #Dotplot3 - action-timestep align
    fig3, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    for i,l in enumerate(actions_data):
        if (len(l)==0):
            continue

        x = range(1,ac_length)
        actions = l[::5]
        actions_2.append(actions)

        pad = len(x)-len(actions)
        x = x[pad:]
        x_2.append(x)

        y = [i] * len(x)

        ax1.scatter(x,y,color = [COLORS[ind] for ind in actions],s=1)

        try:
            spout = (spout_data[i][0]-1)//5+1
            ax1.scatter(pad + spout,i,color ='red',s=1)
            trial_start = (trial_start_data[i][0]-1)//5+1
            ax1.scatter(pad + trial_start,i,color ='red',s=1, marker = 'x')
        except:
            continue

    # ax1.set_title("Actions(orange:lick, green:move, purple:move+lick, black: no action)\n pos rew: {}, move_lick: {}, move: {}, lick: {}, stop: {}".format(train_args.pos_rew, train_args.move_lick, train_args.move, train_args.lick, train_args.stop))
    ax1.get_xaxis().set_visible(False)
    ax1.invert_yaxis()

    #Lineplot3 - action-timestep align
    line_y3 = [[0 for i in range(ac_length)] for j in range(4)]
    all_actions_num = [0 for i in range(ac_length)]
    for i in range(len(x_2)):
        for j in range(len(x_2[i])):
            line_y3[actions_2[i][j]][x_2[i][j]] += 1
            all_actions_num[x_2[i][j]] += 1
    for i in range(4):
        for j in range(ac_length):
            if all_actions_num[j] == 0:
                continue
            line_y3[i][j] /= all_actions_num[j]
    # print(actions_nor)
    for i in range(4):
        ax2.plot(line_y3[i], color = COLORS[i])
    ax2.set_xlabel("Steps(timestep aligned)", fontsize=10)
    ax2.set_ylabel("Actions(averaged)", fontsize=10)

    fig3.suptitle(f"{args.agent} ({algo_name.upper()}) / {args.env} / {date_today}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    args = get_args_licking()
    print(args)

    plot_licking(args)