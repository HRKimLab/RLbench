import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from datetime import date

import pandas as pd 
import matplotlib.pyplot as plt

from options import get_args_licking
from custom_envs import (
    OpenLoopStandard1DTrack, OpenLoopTeleportLong1DTrack, OpenLoopPause1DTrack,
    ClosedLoopStandard1DTrack
)

TEMP_STD_ENV = OpenLoopStandard1DTrack()
TEMP_TEL_ENV = OpenLoopTeleportLong1DTrack()
TEMP_PAU_ENV = OpenLoopTeleportLong1DTrack()
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

    data_path /= "reward_history.pkl"

    licking_data = pd.read_pickle(data_path)
    plt.axvline(x=spout_time, color='r', linestyle='-')
    plt.xlim(260, 320)
    plt.plot(licking_data[-20])
    plt.show()
    # print(licking_data[-50:-1])
    # print(licking_data[-2])
    # plt.plot(licking_data)
    # plt.show()
    # if args.name is not None:
    #     plt.show(block=False)
    #     plt.pause(3)
    #     plt.close()
    #     plt.savefig(f"{args.env}-{args.agent}.png")
    # else:
    #     plt.show()

if __name__ == "__main__":
    args = get_args_licking()
    print(args)

    plot_licking(args)
