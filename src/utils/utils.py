import os
import os.path as p
import json
import random
import logging
from shutil import rmtree
from pathlib import Path
from datetime import date

import numpy as np
import torch
from torch.backends import cudnn
from gym import envs
from sb3_contrib import ARS, QRDQN, TQC, TRPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env, make_atari_env

from custom_envs import MaxAndSkipEnv, OpenLoopStandard1DTrack, OpenLoopTeleportLong1DTrack
from custom_algos import CustomDQN

FLAG_FILE_NAME = "NOT_FINISHED"
ALGO_LIST = {
    "a2c": A2C, "ddpg": DDPG, "dqn": DQN,
    "ppo": PPO, "sac": SAC, "td3": TD3,
    "ars": ARS, "qrdqn": QRDQN, "tqc": TQC, "trpo": TRPO,
    "custom_dqn": CustomDQN
}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def configure_cudnn(debug=False):
    cudnn.enabled = True
    cudnn.benchmark = True
    if debug:
        cudnn.deterministic = True
        cudnn.benchmark = False

def load_json(hp_path):
    with open(hp_path, "r") as f:
        hp = json.load(f)

    #TODO Should be modified
    for k in hp.keys():
        if isinstance(hp[k], list):
            hp[k] = tuple(hp[k])

    return hp

def get_logger():
    os.makedirs("../log", exist_ok=True)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # Info logger
    info_logger = logging.getLogger("info")
    info_fileHandler = logging.FileHandler(f"../log/summary_{date.today().isoformat()}.log", mode='a')
    info_fileHandler.setFormatter(formatter)
    info_logger.setLevel(level=logging.INFO)
    info_logger.addHandler(info_fileHandler)

    # Error logger
    error_logger = logging.getLogger("error")
    error_fileHandler = logging.FileHandler(f"../log/error_{date.today().isoformat()}.log", mode='a')
    error_fileHandler.setFormatter(formatter)
    error_logger.setLevel(level=logging.ERROR)
    error_logger.addHandler(error_fileHandler)

    return info_logger, error_logger

def get_env(env_name, n_env, save_path, seed):
    ENV_LIST = [env_spec.id for env_spec in envs.registry.all()]

    env, eval_env = None, None
    if env_name in ENV_LIST:
        if "ALE" in env_name:
            env = make_atari_env(
                env_name, n_envs=n_env,
                seed=seed, monitor_dir=save_path
            )
            env = VecFrameStack(env, n_stack=4)
            eval_env = make_atari_env(
                env_name, n_envs=1,
                seed=np.random.randint(0, 1000), monitor_dir=None
            )
            eval_env = VecFrameStack(eval_env, n_stack=4)
        else:
            env = make_vec_env(
                env_name, n_envs=n_env,
                seed=seed, monitor_dir=save_path
            )
            eval_env = make_vec_env(
                env_name, n_envs=1,
                seed=np.random.randint(0, 1000), monitor_dir=None
            )
    else:
        try:
            if env_name == "OpenLoopStandard1DTrack":
                env = OpenLoopStandard1DTrack()
                eval_env = OpenLoopStandard1DTrack()
            elif env_name == "OpenLoopTeleportLong1DTrack":
                env = OpenLoopTeleportLong1DTrack()
                eval_env = OpenLoopTeleportLong1DTrack()
            else:
                raise ImportError
            env = MaxAndSkipEnv(env, skip=5) # Mouse can lick about 8 times per second, 40 (frames) / 5 (skipping).
            env = Monitor(env, save_path)
            eval_env = MaxAndSkipEnv(env, skip=5)
        except ImportError:
            raise ValueError(f"Given environment name [{env_name}] does not exist.")
    return env, eval_env

def get_algo(algo_name, env, hp, action_noise, seed):
    if algo_name not in ALGO_LIST:
        raise ValueError(f"Given algorithm name [{algo_name}] does not exist.")

    # Get model
    if action_noise is None:
        model = ALGO_LIST[algo_name](env=env, seed=seed, verbose=0, **hp)
    else:
        model = ALGO_LIST[algo_name](env=env, seed=seed, action_noise=action_noise, verbose=0, **hp)

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
        ex_info = load_json(p.join(data_path, aid, DEP2_CONFIG))

        if agent_info == ex_info:
            agent_id = aid
            data_path = p.join(data_path, agent_id)
            break

    if agent_id is None: # Not found
        agent_id = "a1" if len(agent_list) == 0 \
            else f"a{max(map(lambda x: int(x.split('a')[-1]), agent_list)) + 1}"
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
        session_info = load_json(p.join(data_path, sid, DEP3_CONFIG))
        
        if hp == session_info:
            session_id = sid.lstrip(agent_id)
            data_path = p.join(data_path, sid)
            break

    if session_id is None: # Not found
        session_id = "s1" if len(session_list) == 0 \
            else f"s{max(map(lambda x: int(x.split('s')[-1]), session_list)) + 1}"
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
        data_path = p.join(data_path, seed_list[seed])
        if not p.isfile(p.join(data_path, FLAG_FILE_NAME)):
            already_run = True
        else:
            rmtree(data_path)
            os.mkdir(data_path)
    else:
        seed_id = f"r{len(seed_list) + 1}"
        data_path = p.join(
            data_path,
            agent_id + session_id + seed_id + f"-{seed}/"
        )
        os.mkdir(data_path)

    return data_path, already_run

def clean_data_path(target_path):
    target_path = Path(target_path)
    flag_path = target_path / FLAG_FILE_NAME

    if (not p.isdir(target_path)) or (not p.isfile(flag_path)):
        return

    # Level-1
    rmtree(target_path)

    # Level-2, 3, 4
    for _ in range(3):
        target_path = target_path.parent
        other_exists = [p.isdir(fd) for fd in target_path.iterdir()]
        if any(other_exists):
            return
        rmtree(target_path)

def get_algo_from_agent(agent_name, agent_path):
    policy_cfg_path = Path(agent_path).parent.parent / "policy.json"
    with open(policy_cfg_path, "r") as f:
        policy_cfg = json.load(f)
    algo_name = policy_cfg["algorithm"]
    model = ALGO_LIST[algo_name]

    return algo_name, model
