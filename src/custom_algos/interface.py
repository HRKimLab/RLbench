import os
import json
from typing import Union, Optional, Tuple, Dict, Iterable, Any

import numpy as np
import torch as th
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    ConvertCallback,
    EvalCallback
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.vec_env import VecEnv

from utils import utils
from custom_algos import DQN, C51, QRDQN, MGDQN
from custom_algos.common.config import *

ALGO_LIST = {
    "DQN": DQN,
    "C51": C51,
    "QRDQN": QRDQN,
    "MGDQN": MGDQN
}


class CustomAlgorithm(object):
    def __init__(self, model, seed):
        self.model = model
        self.seed = seed

        self.env = model.env
        self._logger = None
        self._custom_logger = False

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4, # Not used
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "ALGO", # Not used
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True, # Not used
        progress_bar: bool = False,
    ):
        # Setup callback functions
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, eval_log_path, progress_bar)

        # Start training
        callback.on_training_start(locals(), globals())
        self.model.train(callback=callback)

        # End training
        callback.on_training_end()

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None, # Not used
        episode_start: Optional[np.ndarray] = None, # Not used
        deterministic: bool = False,
    ):
        if deterministic:
            return self.model.predict(observation, eps=-1.0), None
        else:
            return self.model.env.action_space.sample(), None

    def get_env(self):
        return self.model.env

    @classmethod
    def load(
        cls,
        algo_cls: str,
        path: str,
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto", # Not used
        custom_objects: Optional[Dict[str, Any]] = None, # Not used
        print_system_info: bool = False, # Not used
        force_reset: bool = True, # Not used
        **kwargs,
    ):
        train_config, algo_config = None, None
        with open(os.path.join(path, "train_cfg.json"), "r") as f:
            train_config = TrainConfig(**json.load(f))
        algo = train_config.algo
        env, _ = utils.get_env(env_name=train_config.env_id, n_env=1, save_path=path, seed=train_config.random_seed)
    
        with open(os.path.join(path, "algo_cfg.json"), "r") as f:
            algo_config = ALGO_CONFIG[algo](**json.load(f))

        model = ALGO_LIST[algo](
            env=env,
            seed=train_config.random_seed,
            save_path=path,
            train_config=train_config,
            algo_config=algo_config
        )
        model.load_model()

        return cls(model=model, seed=train_config.random_seed)

    def save(
        self,
        path: str,
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        pass

    def _init_callback(
        self,
        callback: MaybeCallback,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        progress_bar: bool = False, # Not used
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations; if None, do not evaluate.
            Caution, this parameter is deprecated and will be removed in the future.
            Please use `EvalCallback` or a custom Callback instead.
        :param n_eval_episodes: How many episodes to play per evaluation
        :param n_eval_episodes: Number of episodes to rollout during evaluation.
        :param log_path: Path to a folder where the evaluations will be saved
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Create eval callback in charge of the evaluation
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_path,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                verbose=self.verbose,
            )
            callback = CallbackList([callback, eval_callback])

        callback.init_callback(self)
        return callback

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    @property
    def num_timesteps(self) -> int:
        return self.model.num_timesteps

    """ Not supported """
    def get_vec_normalize_env(self) -> None:
        return None


def convert_sb3_cfg_to_custom_cfg(args, hp, algo, seed):
    base_path = os.path.join("config", "default_custom")

    train_cfg, algo_cfg = None, None
    with open(os.path.join(base_path, "train_configs", f"{algo.lower()}.json"), "r") as f:
        train_cfg = json.load(f)
    with open(os.path.join(base_path, "algo_configs", f"{algo.lower()}.json"), "r") as f:
        algo_cfg = json.load(f)

    # Common (based on DQN, C51, QR-DQN, MGDQN)
    train_cfg["run_name"] = "Exp. 0"
    train_cfg["algo"] = algo
    train_cfg["env_id"] = args.env
    train_cfg["n_envs"] = args.nenv
    train_cfg["state_len"] = 4 if "ALE" in args.env else 1
    train_cfg["frame_skip"] = 1
    train_cfg["random_seed"] = seed
    train_cfg["reward_clipping"] = False #IMPORTANT: Use this with proper value
    # train_cfg["loss_cls"] = 
    # train_cfg["loss_kwargs"] = 
    # train_cfg["optim_cls"] = 
    train_cfg["optim_kwargs"] = {
        "lr": hp["learning_rate"]
    }
    train_cfg["batch_size"] = hp["batch_size"]
    train_cfg["train_step"] = args.nstep
    train_cfg["save_freq"] = args.eval_freq # For convenience
    train_cfg["logging_freq"] = args.eval_freq
    train_cfg["device"] = "auto"
    train_cfg["verbose"] = True

    algo_cfg["policy_kwargs"]["policy_type"] = hp["policy"]
    if hp["policy"] == "MlpPolicy":
        algo_cfg["policy_kwargs"] = {
            "policy_type": "MlpPolicy"
        }
        if hp.get("policy_kwargs", None) and hp["policy_kwargs"].get("net_arch", None):
            algo_cfg["policy_kwargs"]["hidden_sizes"] = hp["policy_kwargs"]["net_arch"]
    elif hp["policy"] == "CnnPolicy":
        algo_cfg["policy_kwargs"] = {"policy_type": "CnnPolicy"}
    else:
        assert ValueError("Invalid policy type.")

    algo_cfg["eps_kwargs"] = {
        "init_eps": hp["exploration_initial_eps"],
        "milestones": int(args.nstep * hp["exploration_fraction"]),
        "target_eps": hp["exploration_final_eps"]
    }
    if hp.get("gamma", None): # MGDQN doesn't have `gamma` attribute
        algo_cfg["discount_rate"] = hp["gamma"]
    algo_cfg["soft_update_rate"] = hp["tau"]
    algo_cfg["buffer_size"] = hp["buffer_size"]
    algo_cfg["learning_starts"] = hp["learning_starts"]
    algo_cfg["train_freq"] = hp["train_freq"]
    algo_cfg["target_update_freq"] = hp["target_update_interval"]

    # Algorithm-specific
    if algo == "DQN":
        pass
    elif algo == "C51":
        if hp.get("policy_kwargs", None) and hp["policy_kwargs"].get("v_min", None):
            algo_cfg["v_min"] = hp["policy_kwargs"]["v_min"]
        if hp.get("policy_kwargs", None) and hp["policy_kwargs"].get("v_max", None):
            algo_cfg["v_max"] = hp["policy_kwargs"]["v_max"]
        if hp.get("policy_kwargs", None) and hp["policy_kwargs"].get("n_atom", None):
            algo_cfg["n_atom"] = hp["policy_kwargs"]["n_atom"]
    elif algo == "QRDQN":
        if hp.get("policy_kwargs", None) and hp["policy_kwargs"].get("n_quant", None):
            algo_cfg["n_quant"] = hp["policy_kwargs"]["n_quant"]
    elif algo == "MGDQN":
        if hp.get("policy_kwargs", None) and hp["policy_kwargs"].get("gamma_min", None):
            algo_cfg["gamma_min"] = hp["policy_kwargs"]["gamma_min"]
        if hp.get("policy_kwargs", None) and hp["policy_kwargs"].get("gamma_max", None):
            algo_cfg["gamma_max"] = hp["policy_kwargs"]["gamma_max"]
        if hp.get("policy_kwargs", None) and hp["policy_kwargs"].get("gamma_n", None):
            algo_cfg["gamma_n"] = hp["policy_kwargs"]["gamma_n"]
        algo_cfg.pop("discount_rate")
    else:
        raise NotImplementedError

    train_cfg = TrainConfig(**train_cfg)
    algo_cfg = ALGO_CONFIG[algo](**algo_cfg)
    return train_cfg, algo_cfg

def init_custom_algo(args, algo, hp, env, seed):
    algo = algo.upper()
    train_cfg, algo_cfg = convert_sb3_cfg_to_custom_cfg(args, hp, algo, seed)
    model = ALGO_LIST[algo](
        env=env,
        seed=seed,
        save_path=args.save_path,
        train_config=train_cfg,
        algo_config=algo_cfg
    )
    return CustomAlgorithm(model=model, seed=seed)

# def load_custom_algo(base_path, env_name, agent_path):
#     file_path = os.path.join(base_path, env_name, agent_path)
#     model = ALGO_LIST[algo].load()
#     level1_path, level2_path, level3_path = agent_path.split('/')
#     with open(os.path.join(base_path,env_name, level1_path, "policy.json"), "r") as f:
#         algo = json.load(f)["algorithm"]

#     with open(os.path.join(file_path, "train_configs", f".json"), "r") as f:
#         train_cfg = json.load(f)
#     with open(os.path.join(file_path, "algo_configs", f"{algo.lower()}.json"), "r") as f:
#         algo_cfg = json.load(f)

#     model = ALGO_LIST[algo](
#         env=
#     )