from math import gamma
import math

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
import logging
import random
import os

class CustomDQN:
    def __init__(
        self,
        env = 'CartPole-v1',
        seed = 0,
        total_timesteps = 10000,
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
        policy = "MlpPolicy",
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
        self.monitor_kwargs = {}
        self.monitor_path = '/home/neurlab-dl1/workspace/RLbench/data/CartPole-v1/a1/a1s1/a1s1r1-0/0.monitor.csv'
        self.monitor_path = '/home/neurlab/hyein/RLbench/data/CartPole-v1/a1/a1s1/a1s1r1-0/0.monitor.csv'
        self.env = Monitor(self.env, filename=self.monitor_path, **self.monitor_kwargs)
        self.q_net = nn.Sequential(
            #input layer=4, hidden layer=256, output layer=2
            #input layer =[]
            nn.Linear(4,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )
        self.q_net_target = nn.Sequential(
            #input layer=4, hidden layer=256, output layer=2
            #input layer =[]
            nn.Linear(4,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )
        self.optimizer = optim.Adam(self.q_net.parameters(), LR)
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


    #replay memory
    def memorize(self, state, action, reward, next_state):
        self.memory.append((
            state,
            action,
            torch.FloatTensor([reward]),
            torch.FloatTensor([next_state])
        ))
    
    def act(self, state):
        #epsilon이 될 수 있는 최소값
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            #max(1)은 각 행의 가장 큰 열 값을 반환
            #dimension이 크기 때문에 값과 index의 tuple 형태로 나옴
            #최대 결과의 두번째 열(1)은 최대 요소의 index
            return self.q_net(state).data.max(1)[1].view(1,1)
        else: #무작위 값 (왼,오) >>> tensor([[0]]) 이런 꼴로 나옴
            return torch.LongTensor([[random.randrange(2)]])
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        #경험이 충분히 쌓일 때부터 학습 진행
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)



        #list to Tensor
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        #모델의 입력으로 states를 제공, 현 상태에서 했던 행동의 가치(q 값)
        #gather는 index로 값 뽑아오는 것
        #self.q_net(states)를 통해 neural network 돌았음 >> shape=(64,2) (왼,오...?) >> ex. tensor([[-0.0284,-0.1595],...,[-0.0430,-0.1887]])
        #actions는 tensor([[0],[0],[1],...,[0]]) 이런 식으로 들어있음 >>shape=(64,1)
        current_q = self.q_net(states).gather(1,actions)

        #에이전트가 보는 행동의 미래 가치
        #datach는 기존 tensor를 복사하지만, gradient 전파가 안되는 tensor가 됨
        #dim=1에서 가장 큰 값을 가져옴

        max_next_q = self.q_net(next_states).detach().max(1)[0]
        # max_next_q = self.q_net_target(next_states).detach().max(1)[0]
        expected_q = rewards + (self.gamma * max_next_q)
        print(current_q.squeeze())
        print(expected_q)
        #행동은 expected_q 따라감, MSE_loss로 오차 계산, 역전파, 신경망 학습
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def target_update(self):
        if self._n_calls % self.target_update_interval == 0 :
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

    def process(self):
        agent = CustomDQN()


        for e in range(1, self.episodes+1):
            #state=[cart position(카트 위치),cart velocity(카트 속도),pole angle(막대기 각도),pole velocity at tip(막대기 끝의 속도)]
            state = self.env.reset()
            steps = 0
            while True:
                self._n_calls += 1
                #action 이전에 환경에 대한 관찰값(observation)을 그림
                self.env.render()

                state = torch.FloatTensor([state]) #현 상태를 Tensor화

                action = agent.act(state)
                #print(action)
          
                #item으로 행동의 번호 추출, step함수에 넣어 값 얻음
                #env.step() -> action 이후 환경에서 얻은 observation(관찰값)
                #action.item() >>> 0 or 1 이렇게 숫자로 나옴
                next_state, reward, done, _ = self.env.step(action.item())

                if done:
                    reward = -100
                
                agent.memorize(state, action, reward, next_state)
                # agent.target_update()
                agent.learn()

                state = next_state
                steps += 1
                
                if done:
                    print("episode:{0} score: {1}".format(e,steps))
                    break
        
        # imageio.mimwrite(os.path.join('/home/neurlab-dl1/hyein/python_study_notes/Hyeeiin/reinforcement_learning/DQN_pytorch/', 'DQNAgnet_cartpole_plot2.gif'), frames, fps=15)
            

        # #model save
        # model = CustomDQN()
        # PATH = r'/home/neurlab-dl1/hyein/python_study_notes/Hyeeiin/reinforcement_learning/DQN_pytorch/'
        # torch.save(model, PATH + 'DQNAgent.pt')
    def set_logger(error_log, info_log):
        return 

model = CustomDQN()
model.process()