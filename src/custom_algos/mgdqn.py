from typing import Union

import gym
import numpy as np
import torch
from gym.wrappers.frame_stack import LazyFrames

from custom_algos.algorithm import ValueIterationAlgorithm
from custom_algos.common.replay_buffer import ReplayBuffer
from custom_algos.common.config import TrainConfig, MGDQNConfig


class MGDQN(ValueIterationAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        seed: int,
        save_path: str,
        train_config: TrainConfig,
        algo_config: MGDQNConfig
    ):
        super().__init__(
            env=env,
            seed=seed,
            save_path=save_path,
            train_config=train_config,
            algo_config=algo_config
        )
        assert isinstance(algo_config, MGDQNConfig), "Given config instance should be a MGDQNConfig class."

        # DQN configurations
        self.gamma_min = algo_config.gamma_min
        self.gamma_max = algo_config.gamma_max
        self.gamma_n = algo_config.gamma_n
        self.gamma_range = torch.linspace(
            algo_config.gamma_min,
            algo_config.gamma_max,
            algo_config.gamma_n,
            dtype=torch.float32
        ).to(self.device)
        self.tau = algo_config.soft_update_rate

        self.learning_starts = algo_config.learning_starts
        self.train_freq = algo_config.train_freq
        self.target_update_freq = algo_config.target_update_freq

        self.memory = ReplayBuffer(size=algo_config.buffer_size)
        self.buffer_cnt = 0
        self.soft_vote = algo_config.soft_vote

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
            next_q_vals = self.target_net(next_obses) # (B, n_act, n_gamma)

            # Action decisions
            if self.soft_vote:
                opt_acts = next_q_vals.mean(dim=-1).argmax(dim=-1).view(-1, 1, 1) # (B, 1, 1)
            else:
                batched_votes = next_q_vals.argmax(dim=1) # (B, gamma_n)
                opt_acts = []
                for votes in batched_votes:
                    opt_acts.append(torch.bincount(votes).argmax().item())
                opt_acts = torch.tensor(opt_acts, device=self.device).view(-1, 1, 1) # (B, 1, 1)

            next_q_vals = next_q_vals.gather(1, opt_acts.expand(-1, 1, self.gamma_n)).squeeze(1) # (B, gamma_n)
            y = rewards.unsqueeze(-1) + \
                (~dones).unsqueeze(-1) * self.gamma_range.view(1, -1) * next_q_vals # ~dones == (1 - dones), (B, gamma_n)
    
        # Get predicted Q-value
        pred = self.pred_net(obses).gather(
            1, actions.view(-1, 1, 1).expand(-1, 1, self.gamma_n)
        ).squeeze() # (B, gamma_n)

        # Forward pass & Backward pass
        self.optimizer.zero_grad()
        loss = self.criterion(pred, y)
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
                q_vals = self.pred_net(obses.to(self.device)) # ((n_envs,) n_act, gamma_n)

                # Action decision
                if self.soft_vote:
                    action = q_vals.mean(dim=-1).argmax(dim=-1).cpu().numpy() # ((n_envs,) )
                else:
                    batched_votes = q_vals.argmax(dim=-2) # ((n_envs,) gamma_n)
                    if self.n_envs == 1:
                        action = torch.bincount(batched_votes).argmax().cpu().item()
                    else:
                        action = []
                        for votes in batched_votes:
                            action.append(torch.bincount(votes).argmax().cpu().item())
                    action = np.asarray(action) # ((n_envs,) )
            if self.n_envs == 1:
                action = np.array([action.item()])
        else:
            action = self.rng.choice(self.n_act, size=(self.n_envs, ))

        return action.squeeze()
