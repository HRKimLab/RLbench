""" Train agents with MATLAB virmen"""

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.noise import (
    NormalActionNoise,
    VectorizedActionNoise
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback
)

from utils import (
    set_seed, configure_cudnn, load_json, get_logger,
    get_env, get_algo, set_data_path, clean_data_path, FLAG_FILE_NAME
)
from utils.options import get_args
from utils.sb3_callbacks import TqdmCallback, LickingTrackerCallback


def train(args):
    """ Train with multiple random seeds """
    print('train')
    info_logger, error_logger = get_logger()

    hp = load_json(args.hp)

    reset_curr_cond_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\reset_curr_cond'
    reset_curr_cond = np.memmap(reset_curr_cond_filename, dtype='uint8',mode='r+', shape=(1, 1))
    reset_curr_cond[:] = np.uint8(0)

    for i, seed in enumerate(args.seed):
    # for i, seed in enumerate(args.nseed):
        set_seed(seed)

        # Get appropriate path by model info
        save_path = args.save_path 
        already_run = False
        if save_path is None:
            save_path, already_run = set_data_path(args.algo, args.env, hp, seed)
        args.save_path = save_path

        # If the given setting has already been executed, save_path will be given as None
        if already_run:
            print(f"[{i + 1}/{args.nseed}] Already exists: '{save_path}', skip to run")
            continue

        # Get env, model
        try:
            env, eval_env = get_env(args.env, args.nenv, save_path, seed)
            action_noise = None
            if args.noise == "Normal":
                assert env.action_space.__dict__.get('n') is None, \
                    "Cannot apply an action noise to the environment with a discrete action space."
                action_noise = NormalActionNoise(args.noise_mean, args.noise_std)
                if args.nenv != 1:
                    action_noise = VectorizedActionNoise(action_noise, args.nenv)
            model = get_algo(args, env, hp, action_noise, seed)
        except KeyboardInterrupt:
            clean_data_path(save_path)
        except Exception as e:
            clean_data_path(save_path)
            info_logger.info("Loading error [ENV: %s] | [ALGO: %s]", args.env, args.algo)
            error_logger.error("Loading error with [%s / %s] at %s", args.env, args.algo, datetime.now(), exc_info=e)
            print(e)
            exit()

        # Train with single seed
        try:
            Path(os.path.join(save_path, FLAG_FILE_NAME)).touch()
            print(f"[{i + 1}/{args.nseed}] Ready to train {i + 1}th agent - RANDOM SEED: {seed}")
            is_licking_task = (args.env in ["OpenLoopStandard1DTrack", "OpenLoopTeleportLong1DTrack", "ClosedLoop1DTrack","ClosedLoop1DTrack_virmen"])
            _train(
                model, args.nstep, is_licking_task,
                eval_env, args.eval_freq, args.eval_eps, args.save_freq, save_path, reset_curr_cond
            )
            del env, model
        except KeyboardInterrupt:
            clean_data_path(save_path)
        except Exception as e:
            clean_data_path(save_path)
            info_logger.info("Train error [ENV: %s] | [ALGO: %s]", args.env, args.algo)
            error_logger.error("Train error with [%s / %s] at %s", args.env, args.algo, datetime.now(), exc_info=e)
            print(e)


def _train(
    model, nstep, is_licking_task,
    eval_env, eval_freq, eval_eps, save_freq, save_path, reset_curr_cond
):
    """ Train with single seed """
    print('_train')
    # Set logger
    logger = configure(save_path, ["csv"])
    model.set_logger(logger)

    # Set callbacks
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=eval_eps,
        eval_freq=eval_freq, 
        log_path=save_path,
        best_model_save_path=save_path,
        deterministic=True,
        verbose=0
    )
    tqdm_callback = TqdmCallback()
    callbacks = [eval_callback, tqdm_callback]
    if save_freq != -1:
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix='rl_model'
        )
        callbacks.append(checkpoint_callback)
    if is_licking_task:
        licking_tracker_callback = LickingTrackerCallback(
            env=model.env,
            save_path=save_path
        )
        callbacks.append(licking_tracker_callback)

    # Training
    start = time.time()
    print(model)
    model.learn(
        total_timesteps=nstep,
        callback=callbacks,
        eval_log_path=save_path
    )
    end = time.time()
    print("FPS {}".format(1/((end-start)/5000)))
    
    os.remove(os.path.join(save_path, FLAG_FILE_NAME))
    model.save(os.path.join(save_path, "info.zip"))
    # imageio.mimwrite('C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\src\\' + str(args.env) + str(args.algo) + '.gif', model.frames, fps=15)
    reset_curr_cond[:] = np.uint8(1)
    print(reset_curr_cond)


if __name__ == "__main__":
    print('__main__')
    args = get_args()
    configure_cudnn()
    print(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} device")
    print("---START EXPERIMENTS---")
    train(args)
