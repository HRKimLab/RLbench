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
from custom_envs import ClosedLoopStandard1DTrack

TEMP_CSTD_ENV = ClosedLoopStandard1DTrack()
FPS = 39.214 
LICK_PER_SEC = 8
MOV_PER_SEC = 4
WINDOW_SIZE = 5
TRACK_LEN = {
    "ClosedLoopStandard1DTrack": TEMP_CSTD_ENV.data.shape[0],
}
SPOUT_TIME = {
    "ClosedLoopStandard1DTrack": TEMP_CSTD_ENV.water_spout
}

def moving_average(data, w=5):
    return np.convolve(data, np.ones(w) / w, mode='valid')

def plot_licking(args):
    """ Plot the behavior (licking) data of mouse agent """

    date_today = date.today().isoformat()
    track_len = TRACK_LEN[args.env.split('_')[0]]
    spout_time = SPOUT_TIME[args.env.split('_')[0]]

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

    licking_data = pd.read_pickle(data_path / "lick_pos.pkl")
    moving_data = pd.read_pickle(data_path / "move_timing.pkl")
    lick_mat = np.zeros((len(licking_data), track_len))

    # Dotplot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    for i, l in enumerate(licking_data):
        x = l[::5]
        y = [i] * len(x)

        ax1.scatter(x, y, color='black', s=1)
        lick_mat[i, x] = 1
    ax1.set_title("Lick")
    ax1.axvline(x=spout_time, color='r', linestyle='-')
    ax1.get_xaxis().set_visible(False)
    ax1.invert_yaxis()

    # Lineplot
    line_y = []
    for i in range(track_len - LICK_PER_SEC):
        half_window = LICK_PER_SEC // 2
        line_y.append(lick_mat[:, i-half_window:i+half_window].sum() / LICK_PER_SEC)
    ax2.plot(line_y)
    ax2.set_ylabel("Lick (licks/s)", fontsize=10)
    ax2.axvline(x=spout_time, color='r', linestyle='-')

    # Speed
    speed_y = []
    mov_mat = np.zeros((track_len, ))
    for i, l in enumerate(moving_data):
        if len(l) != 0:
            mov_mat[np.arange(len(l))] += 1
            # mov_mat[np.cumsum(l)] += 1
    # for i in range(3000 - MOV_PER_SEC):
    #     half_window = MOV_PER_SEC // 2
    #     speed_y.append(mov_mat[:, i-half_window:i+half_window].sum() / MOV_PER_SEC)
    ax3.plot(moving_average(mov_mat))
    ax3.set_xlabel("Position", fontsize=10)
    ax3.set_ylabel("Speed", fontsize=10)
    ax3.axvline(x=spout_time, color='r', linestyle='-')

    # Common
    algo_name, _ = get_algo_from_agent(args.agent, data_path)
    fig.suptitle(f"{args.agent} ({algo_name.upper()}) / {args.env} / {date_today}")
    plt.subplots_adjust(wspace=0, hspace=0)

    if args.name is not None:
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        plt.savefig(f"{args.env}-{args.agent}.png")
    else:
        plt.show()

if __name__ == "__main__":
    args = get_args_licking()
    print(args)

    plot_licking(args)
