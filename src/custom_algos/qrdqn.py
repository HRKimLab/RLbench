from typing import Union

import gym
import numpy as np
import torch
from gym.wrappers.frame_stack import LazyFrames

from custom_algos.algorithm import ValueIterationAlgorithm
from custom_algos.common.replay_buffer import ReplayBuffer
from custom_algos.common.config import TrainConfig, QRConfig


class QRDQN(ValueIterationAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        seed: int,
        save_path: str,
        train_config: TrainConfig,
        algo_config: QRConfig
    ):
        super().__init__(
            env=env,
            seed=seed,
            save_path=save_path,
            train_config=train_config,
            algo_config=algo_config
        )
        assert isinstance(algo_config, QRConfig), "Given config instance should be a QRConfig class."

        # QRDQN configurations
        self.gamma = algo_config.discount_rate
        self.tau = algo_config.soft_update_rate

        self.learning_starts = algo_config.learning_starts
        self.train_freq = algo_config.train_freq
        self.target_update_freq = algo_config.target_update_freq

        self.memory = ReplayBuffer(size=algo_config.buffer_size)
        self.buffer_cnt = 0

        self.n_quant = algo_config.n_quant
        quants = torch.linspace(0.0, 1.0, self.n_quant + 1, dtype=torch.float32).to(self.device)
        self.quants_range = (quants[:-1] + quants[1:]) / 2

    # Update online network with samples in the replay memory. 
    def update_network(self) -> None:
        self.pred_net.train()

        # Do sampling from the buffer
        obses, actions, rewards, next_obses, dones = tuple(map(
            lambda x: torch.from_numpy(x).to(self.device),
            self.memory.sample(self.batch_size)
        )) 
        obses, rewards, next_obses = obses.float(), rewards.float(), next_obses.float()
        actions, rewards = actions.long(), rewards.float()
        if self.reward_clipping:
            rewards = torch.clamp(rewards, -1, 1)

        # obses, next_obses         : (B, state_len, 84, 84)
        # actions, rewards, dones   : (B, )

        # Get q-value from the target network
        with torch.no_grad():
            # Estimated quantiles with target networks
            next_q_quants = self.target_net(next_obses) # (B, n_act, n_quant)
            opt_acts = torch.mean(next_q_quants, dim=-1).argmax(dim=-1).view(-1, 1, 1) # (B, 1, 1)
            est_quants = next_q_quants.gather(
                1, opt_acts.expand(-1, 1, self.n_quant)
            ).squeeze() # (B, n_quant)
            y_quants = rewards.view(-1, 1) + self.gamma * est_quants * (~dones).view(-1, 1) # (B, n_quant)

        # Predicted quantiles with pred. networks
        pred_quants = self.pred_net(obses).gather(
            1, actions.view(-1, 1, 1).expand(-1, 1, self.n_quant)
        ).squeeze() # (B, n_quant)

        # Forward pass & Backward pass
        self.optimizer.zero_grad()
        loss = self.criterion(pred_quants, y_quants, self.quants_range)
        loss.backward()
        self.optimizer.step()

    # Return desired action(s) that maximizes the Q-value for given observation(s) by the online network.
    def predict(
        self,
        obses: Union[list, np.ndarray],
        eps: float = -1.0
    ) -> np.ndarray:
        """
            obses: 
                Training stage : (n_envs, state_len, *, ) or (state_len, *, ) or (*, _)
                Inference stage: (batch_size, state_len, *, ) or (state_len, *, _), or (*, _)
            eps:
                -1.0 at inference stage
        """
        if isinstance(obses, LazyFrames):
            obses = obses[:]
        if isinstance(obses, list):
            obses = np.array(list)
        if isinstance(obses, np.ndarray):
            obses = torch.from_numpy(obses)

        # Epsilon-greedy
        if self.rng.random() >= eps:
            self.pred_net.eval()
            with torch.no_grad():
                action_value = self.pred_net(obses.to(self.device)).mean(dim=-1) # (n_envs, n_actions)
                action = torch.argmax(action_value, dim=-1).cpu().numpy() # (n_envs, )
            if self.n_envs == 1:
                action = np.array([action.item()])
        else:
            action = self.rng.choice(self.n_act, size=(self.n_envs, ))

        return action
