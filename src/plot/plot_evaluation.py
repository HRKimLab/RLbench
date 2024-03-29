# plot evaluation (HyeIn)
import os
from datetime import date
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from options import MAPPER_Y, get_args

def plot_eval(args):
    """ Plot the evaluation rewards on y-axis
    args: user arguments
    """

    date_today = date.today().isoformat()

    data_path = args.data_path
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data' if data_path is None else data_path))

    file_paths = []
    for (root_dir, _, files) in os.walk(os.path.join(data_path, args.env)):
        for file in files:
            if "evaluations.npz" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)

    agent_list = args.agents
    _, y_name = MAPPER_Y[args.y]
    if (args.overwrite == 'y') and (y_name == 'reward') and(args.x == 'timesteps'):
        overwrite = 'y'
    else:
        overwrite = 'n'
        plt.figure(figsize=(10, 6), facecolor=(.94, .94, .94))
    
    total = []
    #if agent name has seed, do not overwrite
    for agent in agent_list:
        for file_path in file_paths:
            if agent in file_path:
                eval_data = np.load(file_path)
                results = eval_data['results']
                results_mean = [sum(results[x])/len(results[x]) for x in range(len(results))]
                total.append(results_mean)
                if overwrite == 'n':
                    plt.plot(eval_data['timesteps']*4, results_mean, label= file_path.split(os.sep)[-2], linestyle = '--')
        if overwrite == 'y':
            zip_seed = list(zip(total[0],total[1],total[2]))
            mean_seed = [np.mean(zip_seed[i]) for i in range(len(zip_seed))]
            plt.plot(eval_data['timesteps']*4, mean_seed , label= agent+'-eval-mean', linestyle = '--')
            total = []
                    
    
    plt.xlabel(args.x)
    plt.ylabel('reward')
    if overwrite == 'y':
        plt.title(f"{args.env} [{args.x}-reward] train & evaluation data {date_today}")
    else:
        plt.title(f"{args.env} [{args.x}-reward] evaluation data {date_today}")

    # Sort legends by name
    #handles, labels = plt.gca().get_legend_handles_labels()
    #labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    #plt.legend(handles, labels)
    plt.legend()

    if args.savefig is None:
        plt.savefig('/nfs/share/figure_repository/result_'+datetime.datetime.now().strftime('%y%m%d_%H%M%S')+'-eval')
    else:
        plt.savefig(args.savefig+'-eval')


if __name__ == "__main__":
    args = get_args()
    print(args)
    plot_eval(args)
    plt.show()
