""" Rendering """
import os
import time
import argparse
from pathlib import Path

from sb3_contrib import QRDQN
from stable_baselines3 import DQN

from utils import get_algo_from_agent
from custom_envs import OpenLoopStandard1DTrack, OpenLoopTeleportLong1DTrack

ENV = {
    "OpenLoopStandard1DTrack": OpenLoopStandard1DTrack,
    "OpenLoopTeleportLong1DTrack": OpenLoopTeleportLong1DTrack
}

def get_render_args():
    parser = argparse.ArgumentParser(description="Required for rendering")
    parser.add_argument(
        '--env', '-E', type=str,
        choices=['OpenLoopStandard1DTrack', 'OpenLoopTeleportLong1DTrack']
    )
    parser.add_argument(
        '--agent', '-A', type=str
    )
    args = parser.parse_args()

    return args

def get_model_path(args):
    model_path = Path(os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data')))
    model_path = model_path / args.env

    try:
        for _ in range(2):
            dir_list = os.listdir(model_path)
            for dir_name in dir_list:
                if dir_name in args.agent:
                    model_path /= dir_name
                    break

        dir_list = os.listdir(model_path)
        for dir_name in dir_list:
            if args.agent in dir_name:
                model_path /= dir_name
                break
    except:
        raise FileNotFoundError("Given agent name is not found.")

    model_path /= "best_model.zip"

    return model_path

def render_single(args):
    model_path = get_model_path(args)
    env = ENV[args.env]()
    _, model_class = get_algo_from_agent(args.agent, model_path.parent)
    model = model_class.load(model_path)

    for _ in range(1):
        total_reward = 0
        state = env.reset()
        while True:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
            time.sleep(.005)
            env.render()
        print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    args = get_render_args()
    print(args)

    render_single(args)
