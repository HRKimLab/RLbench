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

FPS = 39.214 
LICK_PER_SEC = 8
WINDOW_SIZE = 5
TRACK_LEN = {
    "OpenLoopStandard1DTrack": 459,
    "OpenLoopTeleportLong1DTrack": 388
}
SPOUT_TIME = {
    "OpenLoopStandard1DTrack": 335,
    "OpenLoopTeleportLong1DTrack": 227
}

def plot_licking(args):
    """ Plot the behavior (licking) data of mouse agent """

    date_today = date.today().isoformat()
    track_len = TRACK_LEN[args.env]
    spout_time = SPOUT_TIME[args.env]

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

    data_path /= "lick_timing.pkl"

    licking_data = pd.read_pickle(data_path)
    data = np.zeros((len(licking_data), track_len))

    # Dotplot
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    for i, l in enumerate(licking_data):
        x = l[::5]
        y = [i] * len(x)

        ax1.scatter(x, y, color='black', s=1)
        data[i, x] = 1
    ax1.set_title("Lick")
    ax1.axvline(x=spout_time, color='r', linestyle='-')
    ax1.get_xaxis().set_visible(False)
    ax1.invert_yaxis()

    # Lineplot
    line_y = []
    for i in range(track_len - LICK_PER_SEC):
        half_window = LICK_PER_SEC // 2
        line_y.append(data[:, i-half_window:i+half_window].sum() // LICK_PER_SEC)
    ax2.plot(line_y)
    ax2.set_xlabel("Time (40fps)", fontsize=10)
    ax2.set_ylabel("Lick (licks/s)", fontsize=10)
    ax2.axvline(x=spout_time, color='r', linestyle='-')

    algo_name, _ = get_algo_from_agent(args.agent, data_path.parent)
    fig.suptitle(f"{args.agent} ({algo_name.upper()}) / {args.env} / {date_today}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

if __name__ == "__main__":
    args = get_args_licking()
    print(args)

    plot_licking(args)
