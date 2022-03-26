import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import date
from options import get_args

Y_AXIS = {
    'rew': (0, 'reward'),
    'len': (1, 'length'),
    't': (2, 'time') 
}

def mean_plot(args):

    date_today = date.today().isoformat()

    file_paths = []
    for (root_dir, _, files) in os.walk(os.path.join(args.data_path, args.env)):
        for file in files:
            if ".csv" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)

    agent_list = args.agents
    y_idx, y_name = Y_AXIS[args.y]

    dfs = [[]*x for x in range(len(agent_list))]
    
    for i in range(len(agent_list)):
        for file in file_paths:
            if agent_list[i] in file:
                df = pd.read_csv(file,skiprows=1)
                dfs[i].append(df)
                df_concat = pd.concat(dfs[i])

                mean_df = df_concat.groupby(df_concat.index).mean()
                std_df = df_concat.groupby(df_concat.index).std()

        plt.plot(mean_df.iloc[:,y_idx], label = agent_list[i]) 
        plt.fill_between(mean_df.index, (mean_df-std_df).iloc[:,y_idx], (mean_df+std_df).iloc[:,y_idx], alpha = 0.5)
                

    plt.xlabel(args.x)
    plt.ylabel(y_name)
    plt.title(f"{args.env} [{args.x}-{y_name}] {date_today}")

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels)
   

    plt.show()

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    mean_plot(args)

