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

TRACK_LEN = {
    "ClosedLoop1DTrack_virmen": 459
}
SPOUT_TIME = {
    "ClosedLoop1DTrack_virmen": 335
}


def plot_licking(args):
    """ Plot the behavior (licking) data of mouse agent """

    date_today = date.today().isoformat()
    track_len = TRACK_LEN[args.env]
    spout_time = SPOUT_TIME[args.env]



if __name__ == "__main__":
    args = get_args_licking()
    print(args)

    plot_licking(args)