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
    InterleavedOpenLoop1DTrack, SequentialInterleavedOpenLoop1DTrack,
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
    "OpenLoopTeleportLong1DTrack": TEMP_TEL_ENV.data.shape[0],
    "OpenLoopPause1DTrack": TEMP_PAU_ENV.data.shape[0],
    "ClosedLoopStandard1DTrack": TEMP_CSTD_ENV.data.shape[0],
}
SPOUT_TIME = {
    "OpenLoopStandard1DTrack": TEMP_STD_ENV.water_spout,
    "OpenLoopTeleportLong1DTrack": TEMP_TEL_ENV.water_spout,
    "OpenLoopPause1DTrack": TEMP_PAU_ENV.water_spout,
    "ClosedLoopStandard1DTrack": TEMP_CSTD_ENV.water_spout
}

def plot_licking(args):
    """ Plot the behavior (licking) data of mouse agent """

    date_today = date.today().isoformat()
    env_name = args.env.split('_')[0]
    track_len = TRACK_LEN[env_name]
    spout_time = SPOUT_TIME[env_name]

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

    lick_data_path = data_path / "eval" / "lick_timing.pkl"
    td_error_path = data_path / "eval" / "td_error.pkl"
    water_timing_path = data_path / "eval" / "water_timing.pkl"
    q0_path = data_path / "eval" / "q_no_lick.pkl"
    q1_path = data_path / "eval" / "q_lick.pkl"

    lick_timing = pd.read_pickle(lick_data_path)
    td_error = pd.read_pickle(td_error_path)
    water_timing = pd.read_pickle(water_timing_path)
    q0_value = pd.read_pickle(q0_path)
    q1_value = pd.read_pickle(q1_path)
    data = np.zeros((len(lick_timing), 200))

    trial_num = len(lick_timing)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True)
    for i, (lt, wt) in enumerate(zip(lick_timing, water_timing)):
        x = np.array(lt) - wt
        ax1.scatter(x, [i] * len(lt), color='black', s=1)
        data[i, np.round(x).astype(int) + 100] = 1

    td_data = np.zeros((len(td_error), 200))
    for i, (te, wt) in enumerate(zip(td_error, water_timing)):
        x = np.round(np.arange(len(te)) - wt + 100).astype(int)
        td_data[i, x] = te

    q0_data = np.zeros((len(q0_value), 200))
    q1_data = np.zeros((len(q1_value), 200))
    for i, (q0, q1, wt) in enumerate(zip(q0_value, q1_value, water_timing)):
        x = np.round(np.arange(len(q0)) - wt + 100).astype(int)
        x = np.round(np.arange(len(q1)) - wt + 100).astype(int)
        q0_data[i, x] = q0
        q1_data[i, x] = q1

    ax1.scatter(np.array(lt) - wt, [i] * len(lt), color='black', s=1)
    ax1.set_xlim([-70, 30])
    ax1.set_ylabel("Trial #", fontsize=10)
    ax1.axvline(x=0, color='blue', linestyle='--')
    ax1.get_xaxis().set_visible(False)
    ax1.invert_yaxis()

    x = np.arange(200) - 100
    ax2.plot(x, np.mean(data, axis=0))
    ax2.set_xlim([-70, 30])
    ax2.set_ylabel("Lick (licks/s)", fontsize=10)
    ax2.axvline(x=0, color='blue', linestyle='--')

    begin, mid = trial_num // 3, trial_num // 3 * 2
    ax3.plot(x, np.mean(td_data[:begin], axis=0), color="limegreen", label="Begin", alpha=0.5)
    ax3.plot(x, np.mean(td_data[begin:mid], axis=0), color="violet", label="Mid", alpha=0.5)
    ax3.plot(x, np.mean(td_data[mid:], axis=0), color="dodgerblue", label="Later", alpha=0.5)
    ax3.plot(x, np.mean(td_data, axis=0), color="red", label="Whole", alpha=0.5)
    ax3.set_xlim([-70, 30])
    ax3.set_ylabel("TD error", fontsize=10)
    ax3.axvline(x=0, color='blue', linestyle='--')
    ax3.legend()

    ax4.plot(x, np.mean(q0_data[900:1100], axis=0), color="limegreen", label="No lick", alpha=0.5)
    ax4.plot(x, np.mean(q1_data[900:1100], axis=0), color="violet", label="Lick", alpha=0.5)
    ax4.set_xlim([-70, 30])
    ax4.set_xlabel("Time (1/8 sec)", fontsize=10)
    ax4.set_ylabel("Q Value", fontsize=10)
    ax4.axvline(x=0, color='blue', linestyle='--')
    ax4.legend()

    algo_name, _ = get_algo_from_agent(args.agent, lick_data_path.parent.parent)
    fig.suptitle(f"{args.agent} ({algo_name.upper()}) / {args.env} / {date_today}")
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()

if __name__ == "__main__":
    args = get_args_licking()
    print(args)

    plot_licking(args)
