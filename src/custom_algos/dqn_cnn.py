import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm

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
            # nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.Conv2d(3, 16, kernel_size=16, stride=4, padding=0),
            nn.ReLU(),
            # nn.AvgPool2d(2,2),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            # nn.AvgPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 1, padding=0),
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
            # nn.Linear(64*170, 256),
            nn.Linear(180096,256),
            nn.ReLU(),
            # nn.Linear(256,output_num)
            nn.Linear(256,2)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        x = x.view(-1)
        return self.linear(x)

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
        self.env = gym.make('CartPole-v1')
        # self.monitor_kwargs = {}
        # self.monitor_path = '/home/neurlab/yw/RLbench/data/CartPole-v1/a1/a1s1/a1s1r1-0/0.monitor.csv'
        # self.env = Monitor(self.env, filename=self.monitor_path, **self.monitor_kwargs)
        self.timesteps = timesteps
        self.memory = deque([],maxlen=10000)

        self.network = CustomCNN(
            observation_space = self.env.observation_space,
            output_num = self.env.action_space
        )

        self.network_target = CustomCNN(
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
        self.target_update_interval = 30
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
            return torch.LongTensor([[self.network.forward(torch.FloatTensor(obs.copy()).permute(2,0,1)).argmax().item()]])
        else: #무작위value (왼,오) >>> tensor([[0]]) 이런 꼴로 나옴
            return torch.LongTensor([[random.randrange(2)]])

    def memorize(self, obs, action, reward, next_obs):
        self.memory.append((
            torch.FloatTensor([obs]),
            action,
            torch.FloatTensor([reward]),
            torch.FloatTensor([next_obs])
        ))

    def learn(self):   
        if len(self.memory) < self.batch_size:
            return
        #경험이 충분히 쌓일 때부터 학습 진행
        batch = random.sample(self.memory, self.batch_size)
        obss, actions, rewards, next_obss = zip(*batch)



        #list to Tensor
        obss = torch.cat(obss)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_obss = torch.cat(next_obss)

        #모델의 입력으로 obs를 제공, 현 상태에서 했던 행동의 가치(q 값)
        current_q = []
        max_next_q = []
        for i in range(self.batch_size):
            current_q.append(self.network.forward(obss[i].permute(2,0,1))[actions[i]])
            max_next_q.append(self.network_target.forward(next_obss[i].permute(2,0,1)).max())
        # current_q = self.network.forward(obss.permute(0,3,1,2)).gather(1,actions)

        #에이전트가 보는 행동의 미래 가치
        #datach는 기존 tensor를 복사하지만, gradient 전파가 안되는 tensor가 됨
        #dim=1에서 가장 큰 값을 가져옴
        # max_next_q = self.network.forward(next_obss.permute(0,3,1,2)).detach().max(1)[0]
        # max_next_q = self.q_net_target(next_states).detach().max(1)[0]

        current_q = torch.Tensor(current_q)
        current_q.requires_grad_(True)
        max_next_q = torch.Tensor(max_next_q)
        max_next_q.requires_grad_(True)
        expected_q = rewards + (self.gamma * max_next_q)

        #행동은 expected_q 따라감, MSE_loss로 오차 계산, 역전파, 신경망 학습
        loss = F.mse_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learn_noreplay(self, obs, action, reward, next_obs):
        current_q = self.network.forward(torch.Tensor(obs.copy()).permute(2,0,1))[action]
        max_next_q = self.network.forward(torch.Tensor(next_obs.copy()).permute(2,0,1)).max()

        expected_q = reward + (self.gamma * max_next_q)
        expected_q = expected_q.reshape([1,1])
       
        loss = F.mse_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def target_update(self, step):
        if step % self.target_update_interval == 0 :
            polyak_update(self.network.parameters(), self.network_target.parameters(), self.tau)

    def process(self):
        agent = CustomDQN('CartPole-v1')
        # self.env = gym.make('CartPole-v1')
        self.env.reset()
        done = False
        episode = 0
        nsteps = 0


        for step in tqdm(range(1, self.timesteps)):
            if not done:
                nsteps +=1
                obs = self.env.render(mode = 'rgb_array')

                action = agent.act(obs)

                next_state, reward, done, info = self.env.step(action.item())

                next_obs = self.env.render(mode='rgb_array')

                if done:
                    reward = -1

                agent.memorize(obs, action, reward, next_obs)

                agent.learn()
                # agent.learn_noreplay(obs, action, reward, next_obs)
                agent.target_update(step)

            else:
                self.env.reset()
                episode +=1 
                print("episode = {}, score = {}".format(episode, nsteps))
                nsteps = 0
                done = False


    def set_logger(error_log, info_log):
        return 

model = CustomDQN('CartPole-v1')
model.process()