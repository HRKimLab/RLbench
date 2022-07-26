import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', '-E', type=str
    )
    parser.add_argument(
        '--agent', type=str
    )
    parser.add_argument(
        '--data-path', '-S', type=str
    )
    
    args = parser.parse_args()

    return args