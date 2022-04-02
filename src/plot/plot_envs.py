import os
from datetime import date

import pandas as pd
import matplotlib.pyplot as plt

from options import MAPPER_Y, get_args_envs

def plot_envs(args):
    """ Plot the data of different environments on y-axis
    args: user arguments
    """

    date_today = date.today().isoformat()
    
    env_list = args.env
    file_paths = []
    for env in env_list:
        for (root_dir, _, files) in os.walk(os.path.join(args.data_path, env)):
            for file in files:
                if ".csv" in file:
                    file_path = os.path.join(root_dir, file)
                    file_paths.append(file_path)

    print(file_paths)
    y_idx, y_name = MAPPER_Y[args.y]

    for env in env_list:
        bundle = []
        for file in file_paths:
            if env in file:
                if args.agent in file:
                    df = pd.read_csv(file, skiprows=1)
                    bundle.append(df)
        
        df_concat = pd.concat(bundle)  
                
        mean_df = df_concat.groupby(df_concat.index).mean()
        std_df = df_concat.groupby(df_concat.index).std()

        plt.plot(mean_df.iloc[:,y_idx], label = env)
        plt.fill_between(mean_df.index, (mean_df-std_df).iloc[:,y_idx], (mean_df+std_df).iloc[:,y_idx], alpha = 0.5)

    plt.xlabel(args.x)
    plt.ylabel(y_name)
    plt.title(f"{args.env} [{args.x}-{y_name}] {date_today}")

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels)
   

    plt.show()


if __name__ == "__main__":
    args = get_args_envs()
    print(args)

    plot_envs(args)
