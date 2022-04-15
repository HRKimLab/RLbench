""" Options for plotting """

import argparse

# Mapper
MAPPER_Y = {
    'rew': (0, 'reward'),
    'len': (1, 'length'),
    't': (2, 'time') 
}

def str2list(s):
    return list(map(lambda x: x.strip(), s[1:-1].split(',')))

def get_args():
    parser = argparse.ArgumentParser(description="Visualize the training/evaluation process")
    parser.add_argument(
        '--env', '-E', type=str,
        #choices=['LunarLanderContinuous-v2'],
        help="Environment name; Available: (LunarLanderContinuous-v2)"
    )
    parser.add_argument(
        '--agents', '-A', type=str2list,
        default=[], help="Agents list to plot (specify the agent numbers)"
    )
    parser.add_argument(
        '--x', type=str,
        choices=['episode'],
        default='episode', help="x-axis variable"
    )
    parser.add_argument(
        '--y', type=str,
        choices=['rew', 'len', 't', 'e_rew'],
        help="y-axis variable; Available: (rew, len, t, e_rew)"
    )
    parser.add_argument(
        '--data-path', '-S', type=str,
        help="Path of data"
    ) #TODO: Suppose that all of data has been saved on 'data/' path (if data_path is None)


    args = parser.parse_args()

    return args

def get_args_envs():
    parser = argparse.ArgumentParser(description="Visualize the training/evaluation process")
    parser.add_argument(
        '--env', '-E', type=str2list,
        default = [],
        help="Environment name"
    )
    parser.add_argument(
        '--agent', '-A', type=str,
        help="Agent to plot (specify the agent numbers)"
    )
    parser.add_argument(
        '--x', type=str,
        choices=['episode'],
        default='episode', help="x-axis variable"
    )
    parser.add_argument(
        '--y', type=str,
        choices=['rew', 'len', 't', 'e_rew'],
        help="y-axis variable; Available: (rew, len, t, e_rew)"
    )
    parser.add_argument(
        '--data-path', '-S', type=str,
        default="/tmp/sb3-log", help="Path of data"
    ) #TODO: Suppose that all of data has been saved on 'data/' path (if data_path is None)

    args = parser.parse_args()

    return args
    

if __name__ == "__main__":
    args = get_args()

    print(args)
