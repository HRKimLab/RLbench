import os
from datetime import date
from re import L

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json

from options import MAPPER_Y, get_args_envs


def plot_envs(args):
    """ Plot the data of different environments on y-axis
    args: user arguments
    """
    with open('plot/BASELINE.json') as json_file:
        json_data = json.load(json_file)

    
    date_today = date.today().isoformat()
    
    env_list = args.env
    file_paths = []

    get_path = args.data_path
    if get_path is None:
        get_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data'))

    env_list_1 = []
    if env_list == ['ALE']:
        for (root_dir, _, files) in os.walk(os.path.join(get_path, 'ALE')):
            for file in files:
                if "monitor.csv" in file:
                    file_path = os.path.join(root_dir, file)
                    env_list_1.append(file_path.split('/')[-6]+'/'+file_path.split('/')[-5])
                    env_list = list(set(env_list_1))

    for env in env_list:
        for (root_dir, _, files) in os.walk(os.path.join(get_path, env)):
            for file in files:
                if "monitor.csv" in file:
                    file_path = os.path.join(root_dir, file)
                    file_paths.append(file_path)
    
    print(env_list)
    y_idx, y_name = MAPPER_Y[args.y]
    list_normalized_score = []
    for env in env_list:
        bundle = []
        for file in file_paths:
            if env in file:
                if args.agent in file:
                    df = pd.read_csv(file, skiprows=1)
                    bundle.append(df.tail(100))
        
        df_concat = pd.concat(bundle, ignore_index = True)  

        mean_df = df_concat.groupby(df_concat.index).mean()
        mean = mean_df.iloc[:,y_idx].mean()
        std = mean_df.iloc[:,y_idx].std()
        
        normalized_score = 100*(mean - json_data[env]['random'])/(json_data[env][args.normalize] - json_data[env]['random'])
        list_normalized_score.append(normalized_score)

    normalized_df = pd.DataFrame(zip(env_list, list_normalized_score), columns = ['env','dqn'])
    width = 0.4

    normalized_df.sort_values(by='dqn', inplace=True, ascending=True)
    fig, ax = plt.subplots(figsize=(10,6), facecolor=(.94, .94, .94))
    
    y = np.arange(len(env_list))  
    ax.barh(y, normalized_df['dqn'], width, color='skyblue')

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    
    rects = ax.patches
    for rect in rects:
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        space = 5
        ha = 'left'
        if x_value < 0:
            space *= -1
            ha = 'right'
        label = '{:,.0f}'.format(x_value) + '%'
        
        plt.annotate(
            label,                      
            (x_value, y_value),         
            xytext=(space, 0),          
            textcoords='offset points',
            va='center',                
            ha=ha)
   
    ax.set_yticklabels(normalized_df['env'])
    

    plt.yticks(np.arange(min(y), max(y)+1, 1.0))
    plt.xlabel('normalized score')
    plt.title('Normalization to '+args.normalize+' / '+f"{date_today}")
    plt.show()

if __name__ == "__main__":
    args = get_args_envs()
    print(args)

    plot_envs(args)