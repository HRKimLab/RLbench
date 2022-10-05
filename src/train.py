""" Train agents """

import os
from datetime import datetime
from pathlib import Path

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
from utils.sb3_callbacks import (
    TqdmCallback, OpenLoopLickingTrackerCallback,
    InterleavedLickingTrackerCallback, ClosedLoopLickingTrackerCallback
)


def train(args):
    """ Train with multiple random seeds """

    info_logger, error_logger = get_logger()

    hp = load_json(args.hp)
    rewards = (args.pos_rew, args.neg_rew)

    for i, seed in enumerate(args.seed):
    # for i, seed in enumerate(args.nseed):
        set_seed(seed)

        # Get appropriate path by model info
        save_path = args.save_path
        already_run = False
        if save_path is None:
            save_path, already_run = set_data_path(args.algo, args.env, hp, seed, rewards)

        # If the given setting has already been executed, save_path will be given as None
        if already_run:
            print(f"[{i + 1}/{args.nseed}] Already exists: '{save_path}', skip to run")
            continue

        # Get env, model
        try:
            env, eval_env = get_env(args.env, args.nenv, save_path, seed, rewards)
            action_noise = None
            if args.noise == "Normal":
                assert env.action_space.__dict__.get('n') is None, \
                    "Cannot apply an action noise to the environment with a discrete action space."
                action_noise = NormalActionNoise(args.noise_mean, args.noise_std)
                if args.nenv != 1:
                    action_noise = VectorizedActionNoise(action_noise, args.nenv)
            model = get_algo(args.algo, env, hp, action_noise, seed)
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
            print(f"Will be saved at ({save_path})")
            _train(
                model, args.env, args.nstep, 
                eval_env, args.eval_freq, args.eval_eps, args.save_freq, save_path
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
    model, env, nstep, 
    eval_env, eval_freq, eval_eps, save_freq, save_path
):
    """ Train with single seed """

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

    env_candidates = [
        "OpenLoopStandard1DTrack",
        "OpenLoopTeleportLong1DTrack",
        "OpenLoopPause1DTrack",
        "InterleavedOpenLoop1DTrack",
        "ClosedLoopStandard1DTrack"
    ]

    if env in env_candidates[:3]:
        callback = OpenLoopLickingTrackerCallback(
            env=model.env,
            save_path=save_path
        )
        callbacks.append(callback)
    elif env == env_candidates[3]: # Interleaved
        callback = InterleavedLickingTrackerCallback(
            env=model.env,
            n_env=3,
            save_path=save_path
        )
        callbacks.append(callback)
    elif env == env_candidates[4]: # Cloop standard
        callback = ClosedLoopLickingTrackerCallback(
            env=model.env,
            save_path=save_path
        )
        callbacks.append(callback)

    # Training
    model.learn(
        total_timesteps=nstep,
        callback=callbacks,
        eval_log_path=save_path
    )

    os.remove(os.path.join(save_path, FLAG_FILE_NAME))
    model.save(os.path.join(save_path, "info.zip"))


if __name__ == "__main__":
    args = get_args()
    configure_cudnn()

    print(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} device")
    print("---START EXPERIMENTS---")
    train(args)
