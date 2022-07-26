""" Options for training """

import os
import argparse
from argparse import ArgumentTypeError
from datetime import datetime

# Pre-defined random seed
# We won't generate random seeds with random generator in order to
# facilitate an comparing & analyzing between different experiments
SEEDS = [0, 42, 53, 7, 10, 123, 5, 2, 1, 12345]

def validate_args(args):
    """ Validate user arguments """

    if args.env is None:
        raise ArgumentTypeError("Environment name is mandatory field.")
    if args.algo is None:
        raise ArgumentTypeError("Algorithm name is mandatory field.")
    if args.nseed > 10:
        raise ArgumentTypeError("nseed should be less than or equal to 10.")
    if not os.path.exists(args.hp):
        raise ArgumentTypeError("Given configuration file name does not exist.")

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for training")

    # Required
    parser.add_argument(
        '--env', '-E', type=str,
        help="Environment name"
    )
    parser.add_argument(
        '--algo', '-A', type=str,
        choices=["a2c", "ddpg", "dqn", "ppo", "sac", "td3", "trpo", "qrdqn", "custom_dqn"],
        help="Algorithm name"
    )
    parser.add_argument(
        '--hp', '-H', type=str, default=None,
        help="Hyperparameter configuration file name (./config/[FILE].json)"
    )

    # Train
    parser.add_argument(
        '--nseed', type=int, default=3,
        help="Number of experiments with various random seeds, max=10"
    )
    parser.add_argument(
        '--nstep', type=int, default=-1,
        help="Number of timesteps to train"
    )
    parser.add_argument(
        '--nenv', type=int, default=1,
        help="Number of processes for parallel execution (vectorized envs)"
    )
    parser.add_argument(
        '--noise', type=str,
        choices=["Normal"],
        default=None,
        help="Type of action noise"
    )
    parser.add_argument(
        '--noise-mean', type=float,
        default=None,
        help="Mean value of action noise"
    )
    parser.add_argument(
        '--noise-std', type=float,
        default=None,
        help="Std. value of action noise"
    )
    parser.add_argument(
        '--save-freq', type=int, default=-1,
        help="Per timesteps to save checkpoint (default:-1; Do not save checkpoint)"
    )

    # Evaluation
    parser.add_argument(
        '--eval-freq', type=int, default=10000,
        help="Evaluate the agent every n steps"
    )

    parser.add_argument(
        '--eval-eps', type=int, default=5,
        help="Number of episodes to use for evaluation"
    )

    # If the save_path has not been given, all of the data
    # would be saved on the pre-defined directory structure.
    parser.add_argument(
        '--save-path', '-S', type=str, default=None,
        help="Path to save the data"
    )

    # Debug mode (not used now)
    # parser.add_argument('--debug', dest='debug', action='store_true')
    # parser.add_argument('--no-debug', dest='debug', action='store_false')
    # parser.set_defaults(debug=False)

    args = parser.parse_args()

    # Post-processing for arguments
    args.algo == args.algo.lower()
    args.seed = SEEDS[:args.nseed]
    if args.hp is None:
        args.hp = os.path.join(
            os.getcwd(), 
            "config", "default", f"{args.algo}.json"
        )
    else:
        args.hp = os.path.join(
            os.getcwd(),
            "config", f"{args.hp}.json"
        )
    validate_args(args)

    return args

def get_search_args():
    parser = argparse.ArgumentParser(description="Arguments for hyperparameter tuning")

    # Required
    parser.add_argument(
        '--env', '-E', type=str,
        help="Environment name"
    )
    parser.add_argument(
        '--algo', '-A', type=str,
        choices=["a2c", "ddpg", "dqn", "ppo", "sac", "td3", "trpo", "qrdqn"],
        help="Algorithm name"
    )
    parser.add_argument(
        '--nenv', type=int, default=1,
        help="Number of processes for parallel execution (vectorized envs)"
    )

    # Optuna
    parser.add_argument(
        '--name', type=str, default=str(datetime.now()),
        help="Study name"
    )
    parser.add_argument(
        '--nstep', type=int, default=-1,
        help="Number of timesteps to train"
    )
    parser.add_argument(
        '--n-trials', type=int, default=1,
        help="Number of trials for finding the best hyperparameters"
    )
    parser.add_argument(
        '--save-path', '-S', type=str, default='../optuna',
        help="Path to save the data"
    )

    args = parser.parse_args()

    # Post-processing for arguments
    args.algo == args.algo.lower()

    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
