import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

import random
import numpy as np
import math

# nn.Flatten()

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, output_num: int, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]

        # input_channel => color image : 3, blak : 1
        self.cnn = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=5, stride=4, padding=0),
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     n_flatten = self.cnn(
        #         torch.as_tensor(observation_space.sample()[None]).float()
        #     ).shape[1]

        # self.n_flatten

        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.linear = nn.Sequential(
            # nn.Linear(self.n_flatten, 256),
            nn.Linear(64, 256),
            nn.ReLU(),
            # nn.Linear(256,output_num)
            nn.Linear(256,2)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CustomDQN:
    def __init__(
        self,
        env,
        seed = 0,
        timesteps = 5000,
        verbose = 0,
        #hyperparameters
        EPISODES = 150,
        EPS_START = 0.9,
        EPS_END = 0.05,
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
        self.env = gym.make(env)
        self.timesteps = timesteps
        self.memory = deque([],maxlen=10000)

        self.network = CustomCNN(
            observation_space = self.env.observation_space,
            output_num = self.env.action_space
        )

        self.optimizer = optim.Adam(self.network.parameters(), LR)
        self.steps_done = 0 #학습할 때마다 증가
        self.memory = deque(maxlen=10000)
        self.episodes = EPISODES
        self.batch_size = BATCH_SIZE

        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY

        self.gamma = GAMMA
        self._n_calls = 0
        self.target_update_interval = 10
        self.tau = tau
        # self.q_net = nn.Sequential(
        #     nn.Conv2d(3,16, kernel_size = 5, stride = 2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )

        # self.q_net_target = nn.Sequential(
        #     nn.Conv2d(3,16, kernel_size = 5, stride = 2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )

    # def set_network(self):
    #     self.env.reset()
    #     obs = self.env.render(mode = "rgb_array")
    #     obs.shape # (400,600,3)



    def act(self, obs):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            return self.network.forward(torch.Tensor(obs))
        else: #무작위 값 (왼,오) >>> tensor([[0]]) 이런 꼴로 나옴
            return torch.LongTensor([[random.randrange(2)]])


    def learn(self):
        agent = CustomDQN('CartPole-v1')
        done = False


        for step in range(1, self.timesteps):
            if not done:
                obs = self.env.render(mode = 'rgb_array')

                q_values = self.network.forward(torch.Tensor(obs).permute(3,0,1,2))

                action = agent.act(obs)





                
                





model = CustomDQN("CartPole-v1")
model.learn()

