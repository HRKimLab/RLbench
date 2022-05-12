import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import ts2xy

from options import MAPPER_X, MAPPER_Y, get_args

def plot_mean_2(args):
    """ Plot the mean of the data and show standard deviation on y-axis
    args: user arguments
    """

    date_today = date.today().isoformat()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    def tolerant_mean(arrs):
        lens = [len(i) for i in arrs]
        arr = np.ma.empty((np.max(lens),len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[:len(l),idx] = l
        return arr.mean(axis = -1)

    data_path = args.data_path
    if data_path is None:
        data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data'))

    file_paths = []
    for (root_dir, _, files) in os.walk(os.path.join(data_path, args.env)):
        for file in files:
            if "monitor.csv" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)

    
    agent_list = args.agents
    _, x_name = MAPPER_X[args.x]
    _, y_name = MAPPER_Y[args.y]
    y_var_list = []
    
    for color, agent in zip(colors, agent_list):
        bundle = []
        for file in file_paths:
            if agent in file:
                df = pd.read_csv(file, skiprows=1)
                bundle.append(df)
                x_var, y_var = ts2xy(df, x_name)
                y_var_list.append(y_var)
                plt.plot(
                    x_var, y_var,
                    label=file.split('/')[-2],
                    color=None if len(agent_list) == 1 else color,
                    alpha=0.2
                )

        if args.x == 'timesteps':
            # Interpolation -
            l_list = list(map(lambda x: x.l.cumsum(), bundle))
            x_val = set()
            for l in l_list:
                x_val = x_val.union(set(l.values))
            x_val = sorted(list(x_val))
            y_val = []
            for r, l in zip(bundle, l_list):
                y_val.append(np.interp(x_val, l, r.r))
            y_val = np.vstack(y_val)
            y_val = y_val.mean(axis=0)

            plt.plot(x_val,y_val, label=str(agent)+'-mean', color=color)
        
        else:
            y_var_mean = tolerant_mean(y_var_list)
            plt.plot(y_var_mean, label=str(agent)+'-mean', color=color)
                
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

    plot_mean_2(args)
