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
from custom_envs import (
    OpenLoopStandard1DTrack, OpenLoopTeleportLong1DTrack, OpenLoopPause1DTrack,
    InterleavedOpenLoop1DTrack,
    ClosedLoopStandard1DTrack,
)

TEMP_STD_ENV = OpenLoopStandard1DTrack()
TEMP_TEL_ENV = OpenLoopTeleportLong1DTrack()
TEMP_PAU_ENV = OpenLoopPause1DTrack()
TEMP_CSTD_ENV = ClosedLoopStandard1DTrack()
FPS = 39.214 
LICK_PER_SEC = 8
WINDOW_SIZE = 5
TRACK_LEN = {
    "OpenLoopStandard1DTrack": TEMP_STD_ENV.data.shape[0],
    "OpenLoopPause1DTrack": TEMP_PAU_ENV.data.shape[0],
    "OpenLoopTeleportLong1DTrack": TEMP_TEL_ENV.data.shape[0],
    "ClosedLoopStandard1DTrack": TEMP_CSTD_ENV.data.shape[0],
}
SPOUT_TIME = {
    "OpenLoopStandard1DTrack": TEMP_STD_ENV.water_spout,
    "OpenLoopPause1DTrack": TEMP_PAU_ENV.water_spout,
    "OpenLoopTeleportLong1DTrack": TEMP_TEL_ENV.water_spout,
    "ClosedLoopStandard1DTrack": TEMP_CSTD_ENV.water_spout
}

def moving_average(data, w=5):
    return np.convolve(data, np.ones(w) / w, mode='valid')

def plot_licking(args):
    """ Plot the behavior (licking) data of mouse agent """

    date_today = date.today().isoformat()
    track_len = list(TRACK_LEN.values())[:args.nenv]
    spout_time = list(SPOUT_TIME.values())[:args.nenv]

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

    lick_datas = []
    for i in range(args.nenv):
        dp = data_path / f"lick_timing_{i}.pkl"
        d = pd.read_pickle(dp)
        lick_datas.append(d)

    total_len = sum([len(x) for x in lick_datas])
    data = np.zeros((total_len, max(track_len)))

    # Dotplot
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    offset = 0
    cset = ['r', 'g', 'b']
    for n, licking_data in enumerate(lick_datas):
        for i, l in enumerate(licking_data):
            x = l[::5]
            y = [i + offset] * len(x)
            ax1.scatter(x, y, color=cset[n], s=1)
            data[i + offset, x] = 1
        ax1.set_title("Lick")
        ax1.axvline(x=spout_time[n], color='black', linestyle='-')
        offset += len(licking_data)
    ax1.get_xaxis().set_visible(False)
    ax1.invert_yaxis()

    # Lineplot
    offset = 0
    for n in range(args.nenv):
        cur_len = len(lick_datas[n])
        line_y = []
        for i in range(max(track_len) - LICK_PER_SEC):
            half_window = LICK_PER_SEC // 2
            line_y.append(data[offset:offset+cur_len, i-half_window:i+half_window].sum() / LICK_PER_SEC)
        ax2.plot(moving_average(line_y), c=cset[n])
        ax2.set_xlabel("Time (40fps)", fontsize=10)
        ax2.set_ylabel("Lick (licks/s)", fontsize=10)
        ax2.axvline(x=spout_time[n], color='black', linestyle='-')
        offset += cur_len

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
