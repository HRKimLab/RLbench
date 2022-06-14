# The following code refers
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/exp_manager.py

""" Search optimal hyperparamters using optuna """

import os
import glob
import time
import pickle as pkl
from pprint import pprint

import gym
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances
from stable_baselines3 import HerReplayBuffer

from utils import get_env, get_algo
from utils.options import get_search_args
from utils.sb3_callbacks import TrialEvalCallback, TqdmCallback
from utils.hyperparams_opt import HYPERPARAMS_SAMPLER


class HPOptimizer:
    def __init__(self, args):
        self.env = args.env
        self.algo = args.algo
        self.nenv = args.nenv

        self.name = args.name
        self.nstep = args.nstep
        self.n_trials = args.n_trials
        self.is_atari_env = self.is_atari(self.env)

        self.log_path = os.path.join(args.save_path, args.algo)
        self.save_path = os.path.join(
            self.log_path, 
            f"{self.env}_{self.get_latest_run_id(self.log_path, self.env) + 1}"
        )
        self.n_evaluations = max(1, self.nstep // int(1e5))
        print(
            f"Doing {self.n_evaluations} intermediate evaluations for pruning based on the number of timesteps."
            " (1 evaluation every 100k timesteps)"
        )

        ## Not used
        self.n_actions = 0

    def objective(self, trial: optuna.Trial) -> float:
        kwargs = {}
        kwargs.update({"policy": "CnnPolicy" if self.is_atari_env else "MlpPolicy"})

        # Hack to use DDPG/TD3 noise sampler
        trial.n_actions = self.n_actions
        # Hack when using HerReplayBuffer
        trial.using_her_replay_buffer = kwargs.get("replay_buffer_class") == HerReplayBuffer
        if trial.using_her_replay_buffer:
            trial.her_kwargs = kwargs.get("replay_buffer_kwargs", {})
        # Sample candidate hyperparameters

        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial)
        kwargs.update(sampled_hyperparams)

        n_envs = 1 if self.algo == "ars" else self.nenv
        env, eval_env = get_env(self.env, n_envs, save_path=None, seed=None)

        # By default, do not activate verbose output to keep
        # stdout clean with only the trials results

        model = get_algo(self.algo, env, kwargs, action_noise=None, seed=None)

        optuna_eval_freq = int(self.nstep / self.n_evaluations)
        # Account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // self.nenv, 1)
        # Use non-deterministic eval for Atari
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=10,
            eval_freq=optuna_eval_freq,
            deterministic=not self.is_atari_env,
        )
        tqdm_callback = TqdmCallback()

        learn_kwargs = {}

        try:
            model.learn(
                self.nstep,
                callback=[eval_callback, tqdm_callback],
                **learn_kwargs
            )
            # Free memory
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def search(self) -> None:
        """ Search best hyperparams """

        sampler = TPESampler()
        pruner =  MedianPruner(n_warmup_steps=self.n_evaluations // 3)

        try:
            study = optuna.create_study(
                sampler=sampler,
                pruner=pruner,
                study_name=self.name,
                load_if_exists=True,
                direction="maximize",
            )
            study.optimize(self.objective, n_trials=self.n_trials)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.env}_{self.n_trials}-trials-{self.nstep}"
            f"-{int(time.time())}"
        )

        self.log_path = os.path.join(self.save_path, self.algo, report_name)

        # Write report
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f"{self.log_path}.csv")

        # Save python object to inspect/re-use it later
        with open(f"{self.log_path}.pkl", "wb+") as f:
            pkl.dump(study, f)

        # Plot optimization result
        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)

            fig1.show()
            fig2.show()
        except (ValueError, ImportError, RuntimeError):
            pass

    @staticmethod
    def get_latest_run_id(log_path: str, env_id: str) -> int:
        """
        Returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.
        :param log_path: path to log folder
        :param env_id:
        :return: latest run number
        """
        max_run_id = 0
        for path in glob.glob(os.path.join(log_path, env_id + "_[0-9]*")):
            file_name = os.path.basename(path)
            ext = file_name.split("_")[-1]
            if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
                max_run_id = int(ext)
        return max_run_id

    @staticmethod
    def is_atari(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point  # pytype: disable=module-attr
        return "AtariEnv" in str(entry_point)


if __name__ == "__main__":
    args = get_search_args()

    print(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} device")
    print("---START TO FINDING BEST HYPERPARAMS---")
    optimizer = HPOptimizer(args)
    optimizer.search()
