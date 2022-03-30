""" Options for training """

import os
import argparse
from argparse import ArgumentTypeError

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
    parser = argparse.ArgumentParser(description="Visualize the training/evaluation process")

    # Required
    parser.add_argument(
        '--env', '-E', type=str,
        help="Environment name"
    )
    parser.add_argument(
        '--algo', '-A', type=str,
        help="Algorithm name"
    )
    parser.add_argument(
        '--hp', '-H', type=str,
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

    # Evaluation
    parser.add_argument(
        '--eval-freq', type=int, default=10000,
        help="Evaluate the agent every n steps"
    )

    parser.add_argument(
        '--eval-eps', type=int, default=5,
        help="Number of episodes to use for evaluation"
    )

    # If save_path has not been indicated, all of data
    # would be saved on the pre-defined directory structure.
    parser.add_argument(
        '--save-path', '-S', type=str, default=None,
        help="Path to save the data"
    )

    # Debugging mode
    parser.add_argument('--DEBUG', dest='debug', action='store_true')
    parser.add_argument('--NO-DEBUG', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    # Post-processing
    args.seed = SEEDS[:args.nseed]
    args.hp = os.path.join(os.getcwd(), f"config/{args.hp}.json")
    validate_args(args)

    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
