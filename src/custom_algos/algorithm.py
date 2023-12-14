import os
import time
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from importlib import import_module
from collections import deque
from typing import List, Union

import gym
import numpy as np
import torch

from custom_algos.common.config import TrainConfig, DQNConfig, C51Config, QRConfig, MGDQNConfig
from custom_algos.common.policy_networks import get_policy_networks


class RLAlgorithm(ABC):
    def __init__(
        self,
        env: gym.Env,
        seed: int,
        save_path: str,
        train_config: TrainConfig,
        algo_config: Union[DQNConfig, C51Config, QRConfig, MGDQNConfig]
    ):
        # Train configurations
        self.run_name = train_config.run_name
        self.env = env
        self.seed = seed
        self.n_act = self.env.unwrapped.action_space.n if train_config.n_envs == 1 else self.env.unwrapped.action_space[0].n

        self.n_envs = train_config.n_envs
        self.state_len = train_config.state_len
        self.reward_clipping = train_config.reward_clipping

        self.batch_size = train_config.batch_size
        self.train_steps = train_config.train_step
        self.save_freq = train_config.save_freq
        self.logging_freq = train_config.logging_freq

        self.device = torch.device(train_config.device)        
        self.verbose = train_config.verbose

        self.loss_cls, self.loss_kwargs = train_config.loss_cls, train_config.loss_kwargs
        self.optim_cls, self.optim_kwargs = train_config.optim_cls, train_config.optim_kwargs

        # For logging & Save
        self.save_path = save_path

        # Maybe not needed in RLbench
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, "train_cfg.json"), "w") as f:
            f.write(json.dumps(train_config.__dict__, indent=4))
        with open(os.path.join(self.save_path, "algo_cfg.json"), "w") as f:
            f.write(json.dumps(algo_config.__dict__, indent=4))

        # Others
        self.rng = np.random.default_rng(seed)

        """ Attributes below are used for stable-baselines3 integration """
        self.num_timesteps = 0
        self._num_timesteps_at_start = 0
        self.eval_env = None

    @abstractmethod
    def train(self) -> List[int]:
        pass

    # Update online network
    @abstractmethod
    def update_network(self) -> None:
        pass

    # Return desired action(s)
    @abstractmethod
    def predict(
        self,
        obses: Union[list, np.ndarray],
    ) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self) -> None:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass


class ValueIterationAlgorithm(RLAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        seed: int,
        save_path: str,
        train_config: TrainConfig,
        algo_config: Union[DQNConfig, C51Config, QRConfig, MGDQNConfig]
    ):
        super().__init__(
            env=env,
            seed=seed,
            save_path=save_path,
            train_config=train_config,
            algo_config=algo_config
        )

        # Policy networks
        self.pred_net = get_policy_networks(
            algo=train_config.algo,
            state_len=self.state_len,
            n_act=self.n_act,
            n_in=self.env.observation_space.shape[-1],
            n_out=getattr(algo_config, "n_out", -1),
            input_shape=self.env.reset().squeeze(0).shape,
            **algo_config.policy_kwargs
        ).to(self.device)
        self.target_net = deepcopy(self.pred_net).to(self.device)
        self.target_net.eval()

        self.criterion = getattr(
            import_module("custom_algos.common.loss") if self.loss_cls in dir(import_module("custom_algos.common.loss")) \
                else import_module("torch.nn"), self.loss_cls
        )(**self.loss_kwargs)
        self.optimizer = getattr(
            import_module("torch.optim"),
            self.optim_cls
        )(params=self.pred_net.parameters(), **self.optim_kwargs)
        self.eps_scheduler = getattr(
            import_module("custom_algos.common.eps_scheduler"),
            algo_config.eps_cls
        )(**algo_config.eps_kwargs)
        self.eps = self.eps_scheduler.get_eps()

    def train(self, callback=None) -> List[int]:
        episode_deque = deque(maxlen=100)
        episode_infos = []
        start_time = time.time()

        best_reward = float('-inf')

        obs = self.env.reset() # (n_envs, state_len, *)
        if callback:
            callback.on_rollout_start()

        for step in range(self.train_steps // self.n_envs):
            action = self.predict(obs, self.eps) # (n_envs, *)

            # Take a step and store it on buffer
            next_obs, reward, done, infos = self.env.step(action)
            self.add_to_buffer(obs, action, next_obs, reward, done)
            self.num_timesteps += self.n_envs

            if callback:
                callback.update_locals(locals())
                if callback.on_step() is False:
                    raise ValueError("Not supported `callback.on_step() == False` in custom algorithm.")

            # Logging for single environment
            if done:
                if callback:
                    callback.on_rollout_end()
                episode_infos.append(infos[0]["episode"])
                episode_deque.append(infos[0]["episode"]["r"])
                next_obs = self.env.reset()
                if callback:
                    callback.on_rollout_start()

            # Learning if enough timestep has been gone 
            if (self.buffer_cnt >= self.learning_starts) \
                    and (self.buffer_cnt % self.train_freq == 0):
                self.update_network() # Train using replay buffer

            # Periodically copies the parameter of the pred network to the target network
            if step % self.target_update_freq == 0:
                self.update_target()
            obs = next_obs
            self.update_epsilon() # Epsilon-greedy
            
            # Logging
            if episode_deque:
                # Verbose (stdout logging)
                if (step % self.logging_freq == 0):
                    used_time = time.time() - start_time
                    print(f"Step: {step} |",
                        f"100-mean reward: {np.mean(episode_deque):.2f} |",
                        f"Latest reward: {episode_deque[-1]:.2f} |",
                        f"Epsilon: {self.eps:.3f} |",
                        f"Used time: {used_time:.3f}"
                    )

                # Save the model
                if self.verbose and (step % self.save_freq == 0):
                    self.save_model()
                    # Save the best model (roughly best)
                    if episode_deque and episode_deque[-1] > best_reward:
                        self.save_model("best_pred_net.pt", "best_target_net.pt")
                        best_reward = episode_deque[-1]
        if callback:
            callback.on_rollout_end()
        self.env.close()

        return episode_infos

    def add_to_buffer(
        self,
        obs: np.ndarray, # float, (n_envs, state_len, *)
        action: np.ndarray, # int, (n_envs, *)
        next_obs: np.ndarray, # float, (n_envs, state_len, *)
        reward: np.ndarray, # int, (n_envs, *)
        done: np.ndarray, # bool, (n_envs, *)
    ) -> None:
        """ For the compatibility to stable-baselines 3"""
        obs, action, next_obs, reward, done = \
            obs.squeeze(0), action.squeeze(0), next_obs.squeeze(0), reward.squeeze(0), done.squeeze(0)

        if self.n_envs == 1: # Single environment
            self.memory.add(obs, action, reward, next_obs, done)
            self.buffer_cnt += 1
        else: # Vectorized environment
            for i in range(self.n_envs):
                self.memory.add(obs[i], action[i], reward[i], next_obs[i], done[i])
            self.buffer_cnt += self.n_envs

    # Update the target network's weights with the online network's one. 
    def update_target(self) -> None:
        for pred_param, target_param in \
                zip(self.pred_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(
                self.tau * pred_param.data + (1.0 - self.tau) * target_param
            )

    # Update epsilon over training process.
    def update_epsilon(self) -> None:
        self.eps = self.eps_scheduler.step()

    # Save model
    def save_model(
        self,
        pred_net_fname: str = "pred_net.pt",
        target_net_fname: str = "target_net.pt"
    ) -> None:
        torch.save(self.pred_net.state_dict(), os.path.join(self.save_path, pred_net_fname))
        torch.save(self.target_net.state_dict(), os.path.join(self.save_path, target_net_fname))

    # Save model
    def load_model(
        self,
        pred_net_fname: str = "pred_net.pt",
        target_net_fname: str = "target_net.pt"
    ) -> None:
        if pred_net_fname == "pred_net.pt" and target_net_fname == "target_net.pt":
            self.pred_net.load_state_dict(torch.load(
                os.path.join(self.save_path, pred_net_fname),
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            self.target_net.load_state_dict(torch.load(
                os.path.join(self.save_path, target_net_fname),
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
        else:
            self.pred_net.load_state_dict(torch.load(
                pred_net_fname,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            self.target_net.load_state_dict(torch.load(
                target_net_fname,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
