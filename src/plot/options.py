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
        help="Environment name; Available: (LunarLanderContinuous-v2, )"
    )
    parser.add_argument(
        '--agents', '-A', type=str2list,
        default=[], help="Agents list to plot (specify the agent numbers)"
    )
    parser.add_argument(
        '--x', type=str,
        default='episode', help="x-axis variable"
    )
    parser.add_argument(
        '--y', type=str,
        help="y-axis variable; Available: (rew, len, t, e_rew, act_v, )"
    )
    parser.add_argument(
        '--data-path', '-S', type=str,
        default="/tmp/sb3-log", help="Path of data"
    )

    return parser

# if __name__ == "__main__":
#     parser = get_args()
#     args = parser.parse_args()

#     print(args)
