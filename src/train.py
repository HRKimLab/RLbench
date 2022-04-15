import os
from datetime import datetime

import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

from .options import get_args
from .utils import (
    set_seed, configure_cudnn, load_json, get_logger,
    get_env, get_model, set_data_path
)
from .utils.sb3_callbacks import TqdmCallback

def train(args):
    """ Train with multiple random seeds """

    info_logger, error_logger = get_logger()
    hp = load_json(args.hp)

    for i, seed in enumerate(args.seed):
        set_seed(seed)

        # Get appropriate path by model info
        save_path = args.save_path
        already_run = False
        if save_path is None:
            save_path, already_run = set_data_path(args.algo, args.env, hp, seed)

        # Get env, model
        try:
            env, eval_env = get_env(args.env, save_path, seed)
            model = get_model(args.algo, env, hp, seed)
        except Exception as e:
            info_logger.info("Loading error [ENV: %s] | [ALGO: %s]", args.env, args.algo)
            error_logger.error("Loading error with [%s / %s] at %s", args.env, args.algo, datetime.now(), exc_info=e)
            exit()

        # If given setting had already been run, save_path will be given as None
        if already_run:
            print(f"[{i + 1}/{args.nseed}] Already exists: '{save_path}', skip to run")
        else: # Train with single seed
            try:
                print(f"[{i + 1}/{args.nseed}] Ready to train {i + 1}th agent - RANDOM SEED: {seed}")
                _train(
                    model, args.nstep,
                    eval_env, args.eval_freq, args.eval_eps, save_path
                )
                del env, model
            except Exception as e:
                info_logger.info("Train error [ENV: %s] | [ALGO: %s]", args.env, args.algo)
                error_logger.error("Train error with [%s / %s] at %s", args.env, args.algo, datetime.now(), exc_info=e)


def _train(
    model, nstep,
    eval_env, eval_freq, eval_eps, save_path
):
    """ Train with single seed """

    # Set logger
    logger = configure(save_path, ["csv"])
    model.set_logger(logger)

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

    model.learn(
        total_timesteps=nstep,
        callback=[eval_callback, tqdm_callback],
        eval_log_path=save_path
    )
    model.save(os.path.join(save_path, "info.zip"))

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

    print(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} device")
    train(args)
