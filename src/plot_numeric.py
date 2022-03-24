import os
from datetime import date

import pandas as pd
import matplotlib.pyplot as plt

from options import get_args

# Mapper
Y_AXIS = {
    'rew': (0, 'reward'),
    'len': (1, 'length'),
    't': (2, 'time') 
}

def plot_numeric(args):
    """ Plot the numeric data on y-axis
    args: user arguments
    """

    date_today = date.today().isoformat()

    file_paths = []
    for (root_dir, _, files) in os.walk(os.path.join(args.data_path, args.env)):
        for file in files:
            if ".csv" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)

    agent_list = args.agents
    y_idx, y_name = Y_AXIS[args.y]
    for agent in agent_list:
        for file_path in file_paths:
            if (agent in file_path):
                df = pd.read_csv(file_path,skiprows=1)
                plt.plot(df.iloc[:, y_idx], label=agent)

    plt.xlabel(args.x)
    plt.ylabel(y_name)
    plt.title(f"{args.env} [{args.x}-{y_name}] {date_today}")

    # Sort legends by name
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels)

    plt.show()


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    plot_numeric(args)
