""" Multi-screen rendering & saving for comparing multiple agents """
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
import imageio
from PIL import Image
from tqdm import tqdm

from custom_envs import (
    OpenLoopStandard1DTrack,
    OpenLoopTeleportLong1DTrack,
    OpenLoopPause1DTrack,
    ClosedLoopStandard1DTrack,
    MaxAndSkipEnv
)

BASE_PATH = "../data/"

def get_vec_env(env_name, n_stack=4):
    env = make_vec_env(env_name, n_envs = 1)
    env = VecFrameStack(env, n_stack= n_stack)
    return env

def get_mouse_env(n_skip=5):
    env = OpenLoopStandard1DTrack()
    env = MaxAndSkipEnv(env, skip=n_skip)
    return env

def render(env_name, model):
    """ Render how agent interact with environment"""
    # env = get_vec_env(env_name)
    # env = env.envs[0]
    env = get_mouse_env()
    obs = env.reset()
    done = False

    td_errors = []
    td_error = []
    for _ in tqdm(range(5000)):
        action, _ = model.predict(obs, deterministic=True)
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).cuda()
        next_obs, reward, done, _ = env.step(action)
        next_obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).cuda()

        q_value_predict = model.q_net(obs_tensor)[0].detach().cpu()[action]
        q_value_target = model.q_net_target(next_obs_tensor)[0].detach().cpu().max()

        gamma = 0.9
        td_error.append((reward + gamma * q_value_target - q_value_predict).item())

        obs = next_obs
        if done:
            td_errors.append(td_error)
            td_error = []
            obs = env.reset()
            done = False
    
    import pickle
    with open("td_error.pkl", "wb") as f:
        pickle.dump(td_errors, f)

if __name__ == "__main__":
    env_name = "OpenLoopStandard1DTrack"
    model = DQN.load("/home/neurlab/hyein/RLbench/data/OpenLoopStandard1DTrack/a1/a1s1/a1s1r1-0/best_model")

    render(env_name, model)
