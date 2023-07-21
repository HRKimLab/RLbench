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
    time_step_path = data_path / "time_step.pkl"
    us_type_path = data_path / "us_type.pkl"

    date_today = date.today().isoformat()
    # track_len = 100

    shockzone_start_data = pd.read_pickle(shockzone_start_path)
    shockzone_end_data = pd.read_pickle(shockzone_end_path)
    shock_timing_data = pd.read_pickle(shock_timing_path)
    # print(shockzone_start_data)
    # print(shockzone_end_data)
    # print(shock_timing_data)
    time_step_data = pd.read_pickle(time_step_path)
    us_type_data = pd.read_pickle(us_type_path)
    # print(us_type_data)
    # print(time_step_data)


    actions_data = pd.read_pickle(actions_path)
    trial_start_data = pd.read_pickle(trial_start_path)
    # print(trial_start_data)
    START = int((max(sum(trial_start_data,[]))-1)//5 +1)
    # START = 0
    lengths = []
    for i in range(len(actions_data)):
        try:
            lengths.append(int(len(actions_data[i])//5 - trial_start_data[i][0]//5))
        except:
            continue
    length = START + max(lengths) + 2
    # print(trial_start_data)
    dis_list = [[] for i in range(5)]
    actions_split = [[] for i in range(5)]
    trial_start_split = [[] for i in range(5)]
    time_step_split = [[] for i in range(5)]

    
    algo_name, _ = get_algo_from_agent(args.agent, data_path)
    COLORS = ["black", "orange", "green", "darkviolet"]
    US_TYPE_COLORS = ["red", "green", "no" "blue", "darkviolet"]
    

    x_2 = []
    actions_2 = []
    # ac_lengths = [len(actions_data[i])//5 for i in range(len(actions_data))]
    # ac_length = max(ac_lengths) + 2

    #Dotplot1 - action-timestep align
    fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    #Group actions by us_type
    for i,l in enumerate(us_type_data):
        if (len(l)==0):
            us_type = 0
        else:
            us_type = l[0]
        actions_split[us_type].append(actions_data[i])
        trial_start_split[us_type].append(trial_start_data[i])
        time_step_split[us_type].append(time_step_data[i])

    #pad for y
    pad = []
    for h in range(5):
        sum_actions = 0
        i = h-1
        while i>=0:
            sum_actions += len(actions_split[i])
            i -= 1
        pad.append(sum_actions)
    
    for h in range(5):
        if (h==2): #do not use reward 2(air puff) in this task
            continue

        for i,l in enumerate(actions_split[h]):
            if (len(i)==0 or len(trial_start_split[h][i])==0 or len(time_step_data[h][i])==0):
                continue

            x = range(1,length)
            actions = l[::5]
            actions_2.append(actions)

            trial_start = (trial_start_split[h][i][0] -1)//5 +1
            dis = START - trial_start
            # print(dis)
            dis_list[h].append(dis)

            x = x[dis:]
            x = x[:len(actions)]
            x_2.append(x)

            y = [i+pad[h]] * len(x)

            ax1.scatter(x,y,color = [COLORS[ind] for ind in actions],s=1)
            ax1.scatter(dis + trial_start, i+pad[h], color ='pink',s=10, marker = 'x')
            time_step = (time_step_split[h][i][0] -1)//5 +1
            ax1.scatter(dis + time_step, i+pad[h], color ='black',s=5, marker = '|')
        ax1.vlines(x=0, ymin=pad[h], ymax=pad[h+1] if h != 4 else len(actions_data), color = US_TYPE_COLORS[h])
        
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


if __name__ == "__main__":
    args = get_args_licking()
    print(args)

    plot_licking(args)