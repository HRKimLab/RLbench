import os
import os.path as p
import json
import random
from importlib import import_module

import numpy as np
import torch
from torch.backends import cudnn
from gym import envs
from sb3_contrib import ARS, QRDQN, TQC, TRPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

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

def get_env(env_name, save_path, seed):
    ENV_LIST = [env_spec.id for env_spec in envs.registry.all()]

    env = None
    if env_name in ENV_LIST:
        env = make_vec_env(
            env_name, n_envs=1,
            seed=seed, monitor_dir=save_path
        )
        eval_env = make_vec_env(
            env_name, n_envs=1,
            seed=np.random.randint(0, 1000), monitor_dir=save_path
        )
        # env = gym.make(env_name)
        # env = Monitor(env, save_path)
        # env = DummyVecEnv([lambda: env])
    else:
        try:
            env = import_module(f"envs.{env_name}")
        except ImportError:
            raise ValueError(f"Given environment name [{env_name}] does not exist.")
    return env, eval_env

def get_model(algo_name, env, hp, seed):
    ALGO_LIST = {
        "a2c": A2C, "ddpg": DDPG, "dqn": DQN,
        "ppo": PPO, "sac": SAC, "td3": TD3,
        "ars": ARS, "qrdqn": QRDQN, "tqc": TQC, "trpo": TRPO,
    }

    if algo_name not in ALGO_LIST:
        raise ValueError(f"Given algorithm name [{algo_name}] does not exist.")

    # Get model
    model = ALGO_LIST[algo_name](env=env, seed=seed, verbose=0, **hp)

    return model

def set_data_path(algo_name, env_name, hp, seed):
    DEP2_CONFIG = "policy.json"
    DEP3_CONFIG = "hyperparams.json"

    agent_info = {
        "algorithm": algo_name,
        "policy": hp["policy"]
    }

    data_path = p.abspath(p.join(os.getcwd(), os.pardir, 'data'))
    os.makedirs(data_path, exist_ok=True)

    agent_id, session_id = None, None

    # Environment (Depth-1)
    data_path = p.join(data_path, env_name)
    os.makedirs(data_path, exist_ok=True)

    # Agent (Depth-2) - Algorithm, Policy
    agent_list = [x for x in os.listdir(data_path) if p.isdir(p.join(data_path, x))]
    for aid in agent_list:
        ex_info_path = p.join(data_path, aid, DEP2_CONFIG)
        with open(ex_info_path, "r") as f:
            ex_info = json.load(f)

        if agent_info == ex_info:
            agent_id = aid
            data_path = p.join(data_path, agent_id)
            break

    if agent_id is None: # Not found
        agent_id = f"a{len(agent_list) + 1}"
        data_path = p.join(data_path, agent_id)
        if p.exists(data_path):
            raise FileExistsError(
                f"Unexpected directory structure detected, \
                    '{data_path}' already exists."
            )

        os.mkdir(data_path)
        with open(p.join(data_path, DEP2_CONFIG), "w") as f:
            json.dump(agent_info, f, indent=4)

    # Session (Depth-3) - Hyperparameters
    session_list = [x for x in os.listdir(data_path) if p.isdir(p.join(data_path, x))]
    for sid in session_list:
        ex_info_path = p.join(data_path, sid, DEP3_CONFIG)
        with open(ex_info_path, "r") as f:
            session_info = json.load(f)
        
        if hp == session_info:
            session_id = sid.lstrip(agent_id)
            data_path = p.join(data_path, sid)
            break

    if session_id is None: # Not found
        session_id = f"s{len(session_list) + 1}"
        data_path = p.join(data_path, agent_id + session_id)
        if p.exists(data_path):
            raise FileExistsError(
                f"Unexpected directory structure detected, \
                    '{data_path}' already exists."
            )

        os.mkdir(data_path)
        with open(p.join(data_path, DEP3_CONFIG), "w") as f:
            json.dump(hp, f, indent=4)

    # Run (Depth-4) - Random seed
    seed_list = dict(map(
        lambda x: (int(x.split("-")[-1]), x),
        [x for x in os.listdir(data_path) if p.isdir(p.join(data_path, x))]
    ))

    already_run = False
    if seed in seed_list.keys(): # Given setting had already been run
        already_run = True
        data_path = p.join(data_path, seed_list[seed])
    else:
        seed_id = f"r{len(seed_list) + 1}"
        data_path = p.join(
            data_path,
            agent_id + session_id + seed_id + f"-{seed}/"
        )
        os.mkdir(data_path)

    return data_path, already_run
