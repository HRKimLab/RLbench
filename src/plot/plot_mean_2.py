import os
from datetime import date
import sys
import numpy as np
sys.path.append('.')

import pandas as pd
import matplotlib.pyplot as plt
#from stable_baselines3.common.results_plotter import ts2xy

from options import MAPPER_Y, get_args


os.environ['KMP_DUPLICATE_LIB_OK']='True'
def plot_mean_2(args):
    """ Plot the mean of the data and show standard deviation on y-axis
    args: user arguments
    """

    date_today = date.today().isoformat()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    file_paths = []
    for agent in args.agents:
        get_path = args.data_path
        if get_path is None:
            get_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data'))
    
    for (root_dir, _, files) in os.walk(os.path.join(get_path, args.env)):
        for file in files:
            if "0.monitor.csv" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)
   
    agent_list = args.agents
    y_idx, y_name = MAPPER_Y[args.y]

    
    for color, agent in zip(colors, agent_list):
        bundle = []
        for file in file_paths:
            if agent in file:
                df = pd.read_csv(file, skiprows=1)
                bundle.append(df)
                
                if args.x == 'timesteps':
                    x_var = np.cumsum(df.l.values)
                    y_var = df.r.values
                elif args.x == 'episode':
                    x_var = np.arange(len(df))
                    y_var = df.r.values
                elif args.x == 'walltime':
                    # Convert to hours
                    x_var = df.t.values / 3600.0
                    y_var = df.r.values
                
                if len(agent_list) == 1:
                    plt.plot(x_var,y_var,label= file.split('/')[-2], alpha=0.2)
                else:
                    plt.plot(x_var,y_var,label= file.split('/')[-2], color=color, alpha=0.2)
                       
        
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

        plt.plot(x_val,y_val, label = str(agent)+' mean', color=color)
                
        

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
