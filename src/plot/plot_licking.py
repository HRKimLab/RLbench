import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os

from options import get_args

WINDOW_SIZE = 5

def plot_licking(args):
    file_paths = []
    data_path = args.data_path
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data' if data_path is None else data_path))
    for (root_dir, _, files) in os.walk(os.path.join(data_path, args.env)):
        for file in files:
            if "lick_timing.pkl" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)

    agent = args.agents[0]
    for file in file_paths:
        if agent in file:
            obj = pd.read_pickle(file)
            data = np.zeros((len(obj), 459))

            fig, axs = plt.subplots(2, 1)
            for i, l in enumerate(obj):
                x = l[::5]
                y = [i] * len(x)

                axs[0].scatter(x, y, color='black', s=1)
                data[i, x] = 1

            line_y = []
            for i in range(459 - WINDOW_SIZE):
                    # print(data[:, i:i+WINDOW_SIZE])
                line_y.append(data[:, i:i+WINDOW_SIZE].sum().sum() / WINDOW_SIZE)
            axs[1].plot(line_y)

            axs[0].axvline(x=335, color='r', linestyle='-')
            axs[0].invert_yaxis()
            plt.show()


if __name__ == "__main__":
    args = get_args()
    print(args)
    plot_licking(args)
    plt.show()
