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
    
    actions_path = data_path / "actions.pkl"
    trial_start_path = data_path / "trial_start_timing.pkl"
    shockzone_start_path = data_path / "shockzone_start_timing.pkl"
    shockzone_end_path = data_path / "shockzone_end_timing.pkl"
    shock_timing_path = data_path / "shock_timing.pkl"

    date_today = date.today().isoformat()
    # track_len = 100

    shockzone_start_data = pd.read_pickle(shockzone_start_path)
    shockzone_end_data = pd.read_pickle(shockzone_end_path)
    shock_timing_data = pd.read_pickle(shock_timing_path)
    # print(shockzone_start_data)
    # print(shockzone_end_data)
    print(shock_timing_data)


    actions_data = pd.read_pickle(actions_path)
    # print(actions_data)
    lengths = [len(actions_data[i]) for i in range(len(actions_data))]
    # # print(lengths)
    length = int(max(lengths))
    # # print(length)

    trial_start_data = pd.read_pickle(trial_start_path)
    # print(trial_start_data)

    
    algo_name, _ = get_algo_from_agent(args.agent, data_path)
    COLORS = ["black", "orange", "green", "darkviolet"]
    

    x_2 = []
    actions_2 = []
    ac_lengths = [len(actions_data[i])//5 for i in range(len(actions_data))]
    ac_length = max(ac_lengths) + 2
    #Dotplot1 - action-timestep align
    fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
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
            for j in range(len(shock_timing_data[i])):
                shock = (shock_timing_data[i][j]-1)//5+1
                ax1.scatter(pad + shock,i,color ='red',s=1)
            trial_start = (trial_start_data[i][0]-1)//5+1
            ax1.scatter(pad + trial_start,i,color ='pink',s=10, marker = 'x')
            shockzone_start = (shockzone_start_data[i][0]-1)//5+1
            ax1.scatter(pad + shockzone_start,i,color ='blue',s=1, marker = "|") # alpha=.5,
            shockzone_end = (shockzone_end_data[i][0]-1)//5+1
            ax1.scatter(pad + shockzone_end,i,color ='blue',s=1, marker = "|")
        except:
            continue

    # ax1.set_title("Actions(black: no action, yellow: move)\n move: {}, stop: {}, shock: {}".format(0,0,-1000))
    ax1.get_xaxis().set_visible(False)
    ax1.invert_yaxis()

    #Lineplot1 - action-timestep align
    line_y = [[0 for i in range(ac_length)] for j in range(4)]
    all_actions_num = [0 for i in range(ac_length)]
    for i in range(len(x_2)):
        for j in range(len(x_2[i])):
            line_y[actions_2[i][j]][x_2[i][j]] += 1
            all_actions_num[x_2[i][j]] += 1
    for i in range(4):
        for j in range(ac_length):
            if all_actions_num[j] == 0:
                continue
            line_y[i][j] /= all_actions_num[j]
    # print(actions_nor)
    for i in range(4):
        ax2.plot(line_y[i], color = COLORS[i])
    ax2.set_xlabel("Steps(timestep aligned)", fontsize=10)
    ax2.set_ylabel("Actions(averaged)", fontsize=10)

    fig1.suptitle(f"{args.agent} ({algo_name.upper()}) / {args.env} / {date_today}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    x_2 = []
    actions_2 = []
    #Dotplot2 - shockzone 
    fig2, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    for i,l in enumerate(actions_data):
        if (len(l)==0):
            continue
        try:
            x = range(1,ac_length)
            actions = l[::5]

            shockzone_start = (shockzone_start_data[i][0]-1)//5+1
            shockzone_end = (shockzone_end_data[i][0]-1)//5+1

            pad = len(x)-len(actions)
            x = x[pad:]

            start_ind = 0
            end_ind = len(x)-1

            while x[start_ind] < pad + shockzone_start:
                start_ind +=1
            while x[end_ind] > pad + shockzone_end:
                end_ind -= 1
            
            actions = actions[start_ind:end_ind]
            actions_2.append(actions)
            x = x[start_ind:end_ind]
            x_2.append(x)

            y = [i] * len(x)

            ax1.scatter(x,y,color = [COLORS[ind] for ind in actions],s=1)

            ax1.scatter(pad + shockzone_start,i,color ='blue',s=1, marker = "|")
            ax1.scatter(pad + shockzone_end,i,color ='blue',s=1, marker = "|")

            #put some code above this line because below can have some error -> except: continue
            for j in range(len(shock_timing_data[i])):
                shock = (shock_timing_data[i][j]-1)//5+1
                ax1.scatter(pad + shock,i,color ='red',s=1)
            # trial_start = (trial_start_data[i][0]-1)//5+1
            # ax1.scatter(pad + trial_start,i,color ='pink',s=1, marker = '|')
        except:
            continue

    # ax1.set_title("Actions(black: no action, yellow: move)\n move: {}, stop: {}, shock: {}".format(0,0,-1000))
    ax1.get_xaxis().set_visible(False)
    ax1.invert_yaxis()

    #Lineplot2 - shockzone
    line_y2 = [[0 for i in range(ac_length)] for j in range(4)]
    all_actions_num = [0 for i in range(ac_length)]
    for i in range(len(x_2)):
        for j in range(len(x_2[i])):
            line_y2[actions_2[i][j]][x_2[i][j]] += 1
            all_actions_num[x_2[i][j]] += 1
    for i in range(4):
        for j in range(ac_length):
            if all_actions_num[j] == 0:
                continue
            line_y2[i][j] /= all_actions_num[j]
    # print(actions_nor)
    for i in range(4):
        ax2.plot(line_y2[i], color = COLORS[i])
    ax2.set_xlabel("Steps(timestep aligned)", fontsize=10)
    ax2.set_ylabel("Actions(averaged)", fontsize=10)

    fig2.suptitle(f"{args.agent} ({algo_name.upper()}) / {args.env} / {date_today}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    args = get_args_licking()
    print(args)

    plot_licking(args)