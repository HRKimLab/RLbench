""" Options for plotting """
import argparse

# Mapper
MAPPER_X = {
    'timesteps': (0,'timesteps'),
    'episode': (1, 'episodes'),
    'walltime':(2, 'walltime')
}

MAPPER_Y = {
    'rew': (0, 'reward'),
    'len': (1, 'length'),
    't': (2, 'time')
}

def str2list(s):
    return list(map(lambda x: x.strip(), s[1:-1].split(',')))

def get_args():
    parser = argparse.ArgumentParser(description="Plot the target data")
    parser.add_argument(
        '--env', '-E', type=str,
        help="Environment name"
    )
    parser.add_argument(
        '--agents', '-A', type=str2list,
        default=[], help="Agents list to plot (specify the agent numbers)"
    )
    parser.add_argument(
        '--x', type=str,
        choices=['timesteps','episode','walltime'],
        default='timesteps', help="x-axis variable"
    )
    parser.add_argument(
        '--y', type=str,
        choices=['rew', 'len', 't', 'e_rew'],
        help="y-axis variable; Available: (rew, len, t, e_rew)"
    )
    parser.add_argument(
        '--data-path', '-S', type=str,
        help="Path of data"
    )
    parser.add_argument(
        '--mean', type=str,
        choices = ['var','line'],
        default = 'var'
    )
    parser.add_argument(
        '--window_size', type=int,
        default = 1
    )
    parser.add_argument(
        '--overwrite','-O',
        choices = ['y','n'],
        default = 'n',
    )
    parser.add_argument(
        '--savefig', type = str,
        #default = '/nfs/share/figure_repository/result_'
    )
    args = parser.parse_args()

    return args

def get_args_envs():
    parser = argparse.ArgumentParser(description="Plot the target data")
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
    )
    parser.add_argument(
        '--normalize', type=str,
        choices=['human', 'dqn_paper']
    )

    args = parser.parse_args()

    return args

def get_args_licking():
    parser = argparse.ArgumentParser(description="Plot the target data")
    parser.add_argument(
        '--env', '-E', type=str,
        help="Environment name"
    )
    parser.add_argument(
        '--agent', '-A', type=str,
        help="Agent to plot (specify the agent numbers)"
    )
    parser.add_argument(
        '--name', '-N', type=str, default=None,
        help="Name of save file (Default: Not save)"
    )
    parser.add_argument(
        '--nenv', type=int, default=2,
        help="Number of environments (for interleaved environment)"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()

    print(args)
