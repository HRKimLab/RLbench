# plot evaluation (HyeIn)
import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from options import MAPPER_Y, get_args
#get_args 에 overwrite 추가했음

def plot_eval(args):
    """ Plot the evaluation rewards on y-axis
    args: user arguments
    """

    date_today = date.today().isoformat()

    file_paths = []
    for agent in args.agents:
        get_path = args.data_path
        if get_path is None:
            get_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data'))

    for (root_dir, _, files) in os.walk(os.path.join(get_path, args.env)):
        for file in files:
            if "evaluations.npz" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)

    agent_list = args.agents
    y_idx, y_name = MAPPER_Y[args.y]
    if args.overwrite=='y':
        if y_name == 'reward':
            overwrite = 'y'
    else:
        overwrite = 'n'
        fig2, ax2 = plt.subplots(figsize=(10,6), facecolor=(.94, .94, .94))
    for file_path in file_paths:
        for agent in agent_list:
            if agent in file_path:
                eval_data = np.load(file_path)
                results = eval_data['results']
                results_mean = [sum(results[x])/len(results[x]) for x in range(len(results))]
                if overwrite == 'y':
                    plt.plot(eval_data['timesteps'], results_mean, label= file_path.split(os.sep)[-2])
                else:
                    plt.plot(eval_data['timesteps'], results_mean, label= file_path.split(os.sep)[-2])
                    


    plt.xlabel(args.x)
    plt.ylabel('reward')
    if overwrite == 'y':
        plt.title(f"{args.env} [{args.x}-reward] train & evaluation data {date_today}")
    else:
        plt.title(f"{args.env} [{args.x}-reward] evaluation data {date_today}")

    # Sort legends by name
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels)

    plt.show()


if __name__ == "__main__":
    args = get_args()
    print(args)

    plot_numeric(args)
