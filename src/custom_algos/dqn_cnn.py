import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.deepq.policies import MlpPolicy, CnnPolicy

import torch
import torch.nn as nn
from collections import deque

class CustomDQN:
    def __init__(
        self,
        env,
        seed,
        total_timesteps,
        verbose,
        #hyperparameters
        EPISODES,
        EPS_START,
        EPS_END,
        EPS_DECAY = 200,
        GAMMA = 0.8,
        LR = 0.001,
        BATCH_SIZE = 64,

        #from hp
        policy = "CnnPolicy",
        learning_rate = 0.0001,
        buffer_size = 1000000,
        learning_starts = 5000,
        batch_size = 32,
        tau = 1.0,
        gamma = 0.99,
        train_freq = 4,
        gradient_steps = 1,
        target_update_interval = 10000,
        exploration_fraction = 0.1,
        exploration_initial_eps = 1.0,
        exploration_final_eps = 0.05,
        max_grad_norm = 10
    ):
        self.env = 'CartPole-v1'
        self.memory = deque([],maxlen=10000)
        self.q_net = nn.Sequential(
            nn.Conv2d(3,16, kernel_size = 5, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.q_net_target = nn.Sequential(
            nn.Conv2d(3,16, kernel_size = 5, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    #replay buffer
    def memorize(self, state, action, reward, next_state):
        self.memory.append((
            state,
            action,
            torch.FloatTensor([reward]),
            torch.FloatTensor([next_state])
        ))


model = CustomDQN()
env = gym.make('CartPole-v1')
policy = CnnPolicy

