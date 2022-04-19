import os
from datetime import date

import pandas as pd
import matplotlib.pyplot as plt

from options import MAPPER_Y, get_args


def plot_mean_var(args):
    """ Plot the mean of the data and show standard deviation on y-axis
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
            if "0.monitor.csv" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)

    agent_list = args.agents
    y_idx, y_name = MAPPER_Y[args.y]

    for agent in agent_list:
        bundle = []
        for file in file_paths:
            if agent in file:
                df = pd.read_csv(file, skiprows=1)
                bundle.append(df)
        
        df_concat = pd.concat(bundle)  
                
        mean_df = df_concat.groupby(df_concat.index).mean()
        std_df = df_concat.groupby(df_concat.index).std()

        plt.plot(mean_df.iloc[:,y_idx], label = agent)
        plt.fill_between(mean_df.index, (mean_df-std_df).iloc[:,y_idx], (mean_df+std_df).iloc[:,y_idx], alpha = 0.5)

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
