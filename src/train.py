import os
import json
from importlib import import_module

import gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

from options import get_args
from utils import (
    set_seed, configure_cudnn,
    get_env, get_model, set_data_path
)

def train(args):
    """ Train with multiple random seeds """

    for i, seed in enumerate(args.seed):
        set_seed(seed)

        # Get env, model
        env = get_env(args.env)
        eval_env = get_env(args.env)
        model, model_info = get_model(args.algo, env, args.hp, seed)

        # Get appropriate path by model info
        save_path = args.save_path
        if save_path is None:
            save_path = set_data_path(args.env, args.algo, model_info, seed)
            # set_config_files(save_path, model_info)

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

    # Set logger
    logger = configure(save_path, ["stdout", "csv"])
    model.set_logger(logger)

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
