import os
import random

import numpy as np
import torch
from torch.backends import cudnn
from gym import envs
from sb3_contrib import ARS, QRDQN, TQC, TRPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def configure_cudnn(debug):
    cudnn.enabled = True
    cudnn.benchmark = True
    if debug:
        cudnn.deterministic = True
        cudnn.benchmark = False

def get_param_list():
    env_list = [env_spec.id for env_spec in envs.registry.all()]
    algo_list = {
        "a2c": A2C, "ddpg": DDPG, "dqn": DQN,
        "ppo": PPO, "sac": SAC, "td3": TD3,
        "ars": ARS, "qrdqn": QRDQN, "tqc": TQC, "trpo": TRPO,
    }

    return env_list, algo_list

#TODO: Implement
def get_save_path():
    pass