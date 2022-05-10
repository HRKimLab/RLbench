import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import ts2xy

from options import MAPPER_X, MAPPER_Y, get_args


def plot_mean_var(args):
    """ Plot the mean of the data and show standard deviation on y-axis
    args: user arguments
    """

    date_today = date.today().isoformat()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    window_size = 100

    get_path = args.data_path
    if get_path is None:
        get_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data'))
    
    file_paths = []      
    for (root_dir, _, files) in os.walk(os.path.join(get_path, args.env)):
        for file in files:
            if "0.monitor.csv" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)
                
    agent_list = args.agents
    _, x_name = MAPPER_X[args.x]
    _, y_name = MAPPER_Y[args.y]

    for color, agent in zip(colors, agent_list):
        bundle = []
        for file in file_paths:
            if agent in file:
                df = pd.read_csv(file, skiprows=1)
                df['r'] = df.r.rolling(100).mean()
                bundle.append(df)
                x_var, y_var = ts2xy(df, x_name)

        l_list = list(map(lambda x: x.l.cumsum(), bundle))
        x_val = set()
        for l in l_list:
            x_val = x_val.union(set(l.values))
        x_val = sorted(list(x_val))
        y_val = []
        for r, l in zip(bundle, l_list):
            y_val.append(np.interp(x_val, l, r.r))
        y_val = np.vstack(y_val)
        y_val_mean = y_val.mean(axis=0)
        y_val_std = np.std(y_val,axis=0)
         
        #moving_averages = []
        #i = 0
        #while i < len(y_val_mean) - window_size + 1:
            #window = y_val_mean[i : i + window_size]
            #window_average = round(sum(window) / window_size, 2)
            #moving_averages.append(window_average)
            #i += 1
   
        plt.plot(x_val, y_val_mean, label=str(agent)+'-mean', color=color)
        plt.fill_between(x_val, (y_val_mean-y_val_std), (y_val_mean+y_val_std), alpha = 0.5)

    plt.xlabel(args.x)
    plt.ylabel(y_name)
    plt.title(f"{args.env} [{args.x}-{y_name}] {date_today}")

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels)
   

    plt.show()


if __name__ == "__main__":
    args = get_args()
    print(args)

    plot_mean_var(args)
