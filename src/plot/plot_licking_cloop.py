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

    date_today = date.today().isoformat()
    track_len = 100
    spout_data = pd.read_pickle(spout_path)
    # print(spout_data)
    licking_data = pd.read_pickle(licking_path)
    # print(licking_data)
    actions_data = pd.read_pickle(actions_path)
    # print(actions_data)
    data = np.zeros((len(licking_data), track_len))
    dis_list = {}

    # Dotplot - licking
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    for i, l in enumerate(licking_data):

        if (len(spout_data[i]) == 0 or len(l)==0):
            continue
        
        x = l[::5]
        x = [(ele-1)/5+1 for ele in x]

        spout_nor = (spout_data[i][0] -1)//5 +1
        dis = spout_nor - SPOUT
        print(dis)
        dis_list[i] = dis

        x = [int(ele-dis) for ele in x]
        print(x)
        xn = np.array(x)
        # print(xn)
        upper_index = np.where(xn<=100)[0][-1]
        # print(np.where(xn<=100)[0][-1])
        lower_index = np.where(xn>=0)[0][0]
        x = x[lower_index:upper_index]
        y = [i] * len(x)

        ax1.scatter(x, y, color='black', s=1)
        data[i, x] = 1
    ax1.set_title("Lick")
    ax1.axvline(x=SPOUT, color='r', linestyle='-')
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(0,100)
    ax1.invert_yaxis()

    # Lineplot
    line_y = []
    for i in range(track_len - LICK_PER_SEC):
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

    COLORS = ["black", "orange", "green"]
    actions_nor = []

    #Dotplot2 - actions
    fig2, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    for i,l in enumerate(actions_data):
        if (len(spout_data[i]) == 0 or len(l)==0):
            continue

        x = range(1,101)
        actions = l[::5]

        #shift x or actions to match SPOUT and the spout_data
        if (i not in dis_list.keys()):
            continue
        dis = dis_list[i]
        if (dis < 0):
            dis = -dis
            x = x[dis:]
        else:
            actions = actions[dis:]
        
        #match the len of x and actions(colors)
        if (len(x)< len(actions)):
            actions = actions[:len(x)]
        else:
            x = x[:len(actions)]
        
        actions_nor.append(actions)
        y = [i] * len(x)

        ax1.scatter(x,y,color = [COLORS[ind] for ind in actions],s=1)
    ax1.set_title("Actions(orange:lick, green:move, black: no action)")
    ax1.axvline(x=SPOUT, color='r', linestyle='-')
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(0,100)
    ax1.invert_yaxis()   

    #Lineplot2 - actions
    line_y2 = [[0 for i in range(100)] for j in range(3)]
    for i in range(100):
        for j in range(len(actions_nor)):
            try:
                line_y2[actions_nor[j][i]][i] += 1
            except:
                continue
    for i in range(3):
        ax2.plot(line_y2[i], color = COLORS[i])
    ax2.set_xlabel("Steps", fontsize=10)
    ax2.set_ylabel("Lick (licks/s)", fontsize=10)
    ax2.axvline(x=SPOUT, color='r', linestyle='-')

    fig2.suptitle(f"{args.agent} ({algo_name.upper()}) / {args.env} / {date_today}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    args = get_args_licking()
    print(args)

    plot_licking(args)