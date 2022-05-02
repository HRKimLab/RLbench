import os
from datetime import date
from re import L

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from options import MAPPER_Y, get_args_envs

BASELINE = {
    "ALE/Boxing-v5": {
        "random": 0.1,
        "linear": 44,
        "sarsa": 9.8,
        "human": 4.3,
        "dqn_paper": 71.8
    },
    "ALE/Qbert-v5": {
        "random": 163.9,
        "linear": 613.5,
        "sarsa": 960.3,
        "human": 13455,
        "dqn_paper": 10596
    },
    "ALE/Hero-v5":{
        "random": 1027,
        "linear": 6459,
        "sarsa": 7295,
        "human": 25763,
        "dqn_paper": 19950
    },
    "ALE/Breakout-v5": {
        "random": 1.7,
        "linear": 5.2,
        "sarsa": 6.1,
        "human": 31.8,
        "dqn_paper": 401.2
    },
    "ALE/Asterix-v5": {
        "random": 210,
        "linear": 987.3,
        "sarsa": 1332,
        "human": 8503,
        "dqn_paper": 6012
    },
    "ALE/IceHockey-v5": {
        "random": -11.2,
        "linear": -9.5,
        "sarsa": -3.2,
        "human": 0.9,
        "dqn_paper": -1.6
    },
    "ALE/StarGunner-v5": {
        "random": 664,
        "linear": 1070,
        "sarsa": 9.4,
        "human": 10250,
        "dqn_paper": 57997
    },
    "ALE/Robotank-v5":{
        "random": 2.2,
        "linear": 28.7,
        "sarsa": 12.4,
        "human": 11.9,
        "dqn_paper": 51.6
    },
    "ALE/Atlantis-v5":{
        "random": 12850,
        "linear": 62687,
        "sarsa": 852.9,
        "human": 29028,
        "dqn_paper": 85641
    }

}
def plot_envs(args):
    """ Plot the data of different environments on y-axis
    args: user arguments
    """
    
    date_today = date.today().isoformat()
    
    env_list = args.env
    file_paths = []

    get_path = args.data_path
    if get_path is None:
        get_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data'))

    for env in env_list:
        for (root_dir, _, files) in os.walk(os.path.join(get_path, env)):
            for file in files:
                if "monitor.csv" in file:
                    file_path = os.path.join(root_dir, file)
                    file_paths.append(file_path)
    
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
        
        normalized_score = 100*(mean - BASELINE[env]['random'])/(BASELINE[env][args.normalize] - BASELINE[env]['random'])
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
