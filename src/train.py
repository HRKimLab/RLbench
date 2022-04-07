import os
import zipfile

import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

from options import get_args
from utils import (
    set_seed, configure_cudnn,
    get_env, get_model, set_data_path
)
from utils.sb3_callbacks import TqdmCallback

def train(args):
    """ Train with multiple random seeds """

    for i, seed in enumerate(args.seed):
        set_seed(seed)

        # Get env, model
        env = get_env(args.env)
        model, model_info = get_model(args.algo, env, args.hp, seed)

        # Get appropriate path by model info
        save_path = args.save_path
        already_run = False
        if save_path is None:
            save_path, already_run = set_data_path(args.algo, args.env, model_info, seed)

        # If given setting had already been run, save_path will be given as None
        if already_run:
            print(f"[{i + 1}/{args.nseed}] Already exists: '{save_path}', skip to run")
        else: # Train with single seed
            print(f"[{i + 1}/{args.nseed}] Ready to train {i + 1}th agent - RANDOM SEED: {seed}")
            _train(
                model, args.nstep,
                args.eval_freq, args.eval_eps, save_path
            )

        del env, model

def _train(
    model, nstep,
    eval_freq, eval_eps, save_path
):
    """ Train with single seed """

    # Set logger
    logger = configure(save_path, ["csv"])
    model.set_logger(logger)

    #TODO: Sophisticate the evaluation process (eval_eps)
    eval_callback = EvalCallback(
        model.get_env(),
        n_eval_episodes=eval_eps,
        eval_freq=eval_freq, 
        log_path=f"{save_path}/",
        best_model_save_path=f"{save_path}/",
        deterministic=True,
        verbose=0
    )
    tqdm_callback = TqdmCallback()

    model.learn(
        total_timesteps=nstep,
        callback=[eval_callback, tqdm_callback],
        eval_log_path=save_path
    )
    model.save(save_path)

    # Save the logging files
    with zipfile.ZipFile(f"{save_path}.zip", 'r') as zip_ref:
        zip_ref.extractall(save_path)
    os.remove(f"{save_path}.zip")

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
