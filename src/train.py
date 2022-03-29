import os
import json
from importlib import import_module

import gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

from options import get_args
from utils import (
    set_seed, configure_cudnn, get_param_list, get_save_path
)

def get_env(env_name, env_list):
    env = None
    if env_name in env_list:
        env = gym.make(env_name)
    else:
        try:
            env = import_module(f"envs.{env_name}")
        except ImportError:
            raise ValueError(f"Given environment name [{env_name}] does not exist.")
    return env

def get_model(algo_name, algo_list, env, seed, hp):
    model = None, None
    if algo_name not in algo_list:
        raise ValueError(f"Given algorithm name [{algo_name}] does not exist.")
    model = algo_list[algo_name](env=env, seed=seed, verbose=1, **hp)
    return model

def train(args):
    """ Train with multiple random seeds """

    # Get available environments/algorithms list
    env_list, algo_list = get_param_list()

    # Load hyperparameters
    hp = None
    with open(args.hp, "r") as f:
        hp = json.load(f)

    ## Train
    for i, seed in enumerate(args.seed):
        # Util / Log
        set_seed(seed)
        save_path = args.save_path
        if save_path is None:
            save_path = get_save_path(args.env, args.algo, seed)
        logger = configure(save_path, ["stdout", "csv"])

        # Model, Env
        env = get_env(args.env, env_list)
        eval_env = get_env(args.env, env_list)
        model = get_model(args.algo, algo_list, env, seed, hp)
        model.set_logger(logger)

        # Train with single seed
        print(f"[{i + 1}/{args.nseed}] Ready to train {i + 1}th agent - RANDOM SEED: {seed}")
        _train(
            model, args.nstep,
            eval_env, args.eval_freq, args.eval_eps, save_path
        )

        del env, eval_env, model

def _train(
    model, nstep,
    eval_env, eval_freq, eval_eps, save_path
):
    """ Train with single seed """

    #TODO: Sophisticate the evaluation process (eval_eps)
    eval_callback = EvalCallback(
        eval_env, eval_freq=eval_freq, deterministic=False, render=False
    )

    model.learn(total_timesteps=nstep, callback=eval_callback)
    model.save(save_path)

def render(env, model, nstep):
    """ Render how agent interact with environment"""
    obs = env.reset()
    for i in range(nstep):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    args = get_args()
    configure_cudnn(args.debug)

    train(args)
