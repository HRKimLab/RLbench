import os
import os.path as p
import json
import random
from importlib import import_module

import gym
import numpy as np
import torch
from torch.backends import cudnn
from gym import envs
from sb3_contrib import ARS, QRDQN, TQC, TRPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.save_util import data_to_json

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

def get_env(env_name):
    ENV_LIST = [env_spec.id for env_spec in envs.registry.all()]

    env = None
    if env_name in ENV_LIST:
        env = gym.make(env_name)
    else:
        try:
            env = import_module(f"envs.{env_name}")
        except ImportError:
            raise ValueError(f"Given environment name [{env_name}] does not exist.")
    return env

def get_model(algo_name, env, hp, seed):
    ALGO_LIST = {
        "a2c": A2C, "ddpg": DDPG, "dqn": DQN,
        "ppo": PPO, "sac": SAC, "td3": TD3,
        "ars": ARS, "qrdqn": QRDQN, "tqc": TQC, "trpo": TRPO,
    }

    def _trim_model_info(d, targets=[":serialized:", "__doc__"]):
        """ Trim the model information (Drop the unnecessary info) """
        for v1 in list(d.values()):
            if isinstance(v1, dict):
                for k2, v2 in list(v1.items()):
                    # Remove targets from keys of depth-2 on dictionary
                    for target in targets:
                        if target in k2:
                            del v1[target]
                    # Remove keys in which the address is in value
                    if isinstance(v2, str) and ("at 0x" in v2):
                        del v1[k2]

    if algo_name not in ALGO_LIST:
        raise ValueError(f"Given algorithm name [{algo_name}] does not exist.")

    # Load hyperparameters
    with open(hp, "r") as f:
        hp = json.load(f)

    # Get model
    model = ALGO_LIST[algo_name](env=env, seed=seed, verbose=1, **hp)

    # Get model information
    model_info = model.__dict__.copy()
    state_dicts_names, torch_variable_names = model._get_torch_save_params()
    all_pytorch_variables = state_dicts_names + torch_variable_names
    exclude = set(model._excluded_save_params())

    for torch_var in all_pytorch_variables:
        # We need to get only the name of the top most module as we'll remove that
        var_name = torch_var.split(".")[0]
        # Any params that are in the save vars must not be saved by data
        exclude.add(var_name)

    for param_name in exclude:
        model_info.pop(param_name, None)

    model_info = json.loads(data_to_json(model_info))
    _trim_model_info(model_info)

    return model, model_info

def set_data_path(env_name, model_info, seed):
    DEP2_CONFIG = "policy.json"
    DEP3_CONFIG = "hyperparams.json"

    def _sort_dict(obj):
        if isinstance(obj, dict):
            return sorted((k, _sort_dict(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return sorted(_sort_dict(x) for x in obj)
        else:
            return obj

    def _is_same_dict(dict1, dict2):
        return _sort_dict(dict1) == _sort_dict(dict2)

    agent_info = _sort_dict(model_info["policy_class"]["__module__"])
    data_path = p.abspath(p.join(os.getcwd(), os.pardir, 'data'))
    os.makedirs(data_path, exist_ok=True)

    agent_id, session_id = None, None

    # Environment (Depth-1)
    data_path = p.join(data_path, env_name)
    os.makedirs(data_path, exist_ok=True)

    # Agent (Depth-2) - Algorithm, Policy
    agent_list = os.listdir(data_path)
    for aid in agent_list:
        ex_info_path = p.join(data_path, aid, DEP2_CONFIG)
        with open(ex_info_path, "r") as f:
            ex_info = _sort_dict(json.load(f))

        if _is_same_dict(agent_info, ex_info):
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
    session_list = [x for x in os.listdir(data_path) if x != DEP2_CONFIG]
    for sid in session_list:
        ex_info_path = p.join(data_path, sid, DEP3_CONFIG)
        with open(ex_info_path, "r") as f:
            session_info = _sort_dict(json.load(f))
        
        if _is_same_dict(model_info, session_info):
            session_id = sid
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
            json.dump(model_info, f, indent=4)

    # Run (Depth-4) - Random seed
    seed_list = dict(map(
        lambda x: (int(x.split("-")[-1]), x),
        [x for x in os.listdir(data_path) if x != DEP3_CONFIG]
    ))

    already_run = False
    if seed in seed_list.keys(): # Given setting had already been run
        already_run = True
        data_path = p.join(data_path, seed_list[seed])
    else:
        seed_id = f"r{len(seed_list) + 1}"
        data_path = p.join(
            data_path,
            agent_id + session_id + seed_id + f"-{seed}"
        )

    return data_path, already_run
