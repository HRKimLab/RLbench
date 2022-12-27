import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import math
import random
import os
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: int, output_num: int, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.output_num = output_num

        #(1080,1920,3)
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 128, kernel_size=11, stride=3, padding=1), #color image => input channel :3 
        #     nn.ReLU(),
        #     nn.AvgPool2d(2,2),
        #     nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=1),
        #     nn.ReLU(),
        #     # nn.AvgPool2d(2,2),
        #     # nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding=1),
        #     # nn.ReLU(),
        #     # nn.Flatten()
        # )
        #(108,192,3)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=3, padding=1), #color image => input channel :3 
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            # nn.AvgPool2d(2,2),
            # nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding=1),
            # nn.ReLU(),
            # nn.Flatten()
        )

        #(1080,1920,3)
        # self.linear = nn.Sequential(
        #     nn.Linear(128*56109,128),
        #     nn.ReLU(),
        #     nn.Linear(128,64),
        #     nn.ReLU(),
        #     nn.Linear(64,self.output_num)
        # )
        #(540,960,3)
        self.linear = nn.Sequential(
            nn.Linear(27840,64),
            nn.ReLU(),
            nn.Linear(64,self.output_num)
        )

    def forward(self, state: torch.tensor) -> torch.tensor:
        x = self.cnn(state)
        x = x.view(-1)
        return self.linear(x)

class VirmenCNN:
    def __init__(
        self,
        env = "virmen",
        seed = 0,
        nstep = 2000,
        verbose = 0,
        #hyperparameters
        # EPISODES = 15,
        EPS_START = 0.9,
        EPS_END = 0.05,
        EPS_DECAY = 200,
        LR = 0.1,
        tau = 0.1,
        gamma = 0.99
    ):
        self.nstep = nstep
        # self.EPISODES = EPISODES
        self.episode = 0
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0 #학습할 때마다 증가

        self.FRAME_STEP = 100 #update rate of policy
        self.LR = LR
        self.tau = tau
        self.gamma = gamma

        self.licking_cnt = 0
        self.lick_pos_eps = []
        self.lick_pos = []
        self.reward_set = []
        self.reward_set_eps = []

        #network-(1080,1920,3)
        # self.network = CustomCNN(
        #     observation_space = (1080,1920,3),
        #     output_num = 3
        # )

        # self.network_target = CustomCNN(
        #     observation_space = (1080,1920,3),
        #     output_num = 3
        # )
        #(540, 960, 3)
        self.network = CustomCNN(
            observation_space = (108,192,3),
            output_num = 3
        )

        self.network_target = CustomCNN(
            observation_space = (108,192,3),
            output_num = 3
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.network.parameters(), self.LR)

        #flag
        self.img_flag = np.uint8([0]) # 0 for false (initialize)
        self.rew_flag = np.uint8([0])
        self.action_flag = np.uint8([0]) # 1 for true (initialize)
        self.action = np.uint8([0]) #default action

        #file memmap
        image_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_file'
        image_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_flag'
        reward_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_flag'
        reward_mem_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_mem'
        action_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_flag'
        action_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_mem'

        self.img_mem = np.memmap(image_filename, dtype='uint8',mode='r+', shape=(1080, 1920, 3))
        self.img_flag_mem = np.memmap(image_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.rew_flag_mem = np.memmap(reward_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))   
        self.rew_mem = np.memmap(reward_mem_filename, dtype='uint8',mode='r+', shape=(1, 1))   
        self.action_flag_mem = np.memmap(action_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.action_mem = np.memmap(action_filename, dtype='uint8',mode='r+', shape=(1, 1))

        #initialize
        self.action_mem[:] = self.action[:] #default
        self.rew_flag_mem[:]=self.rew_flag[:]
        self.img_flag_mem[:] = self.img_flag[:]
        self.action_flag_mem[:] = self.action_flag[:]

        #black image
        # self.zeros = np.zeros(shape = (1080,1920,3), dtype = np.uint8)
        self.zeros = np.zeros(shape = (108,192,3), dtype = np.uint8)

        self.pos_rew = 10
        self.neg_rew = -5

    #from model find action
    def act(self, state):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            net_out = self.network.forward(torch.tensor(state.copy(), device=self.device).permute(2,0,1).float()).float()
            out = torch.tensor(net_out.argmax().item(), device=self.device).long()
            return out.view(-1, 1)
        else: #무작위value >>> tensor([[0]]) 이런 꼴로 나옴
            return torch.randint(low=0, high=3, size=(1, 1), device=self.device)

    #RESET
    def reset(self):
        #get image
        while (self.img_flag_mem != np.uint8([1])):
            continue
        state_image = self.img_mem
        self.img_flag[:] = np.uint8([0]) #make it false (after reading img frame)

        image_reshape = np.reshape(state_image, (3,1920,1080))
        image_permute = image_reshape.transpose((2,1,0))
        image_resize = cv2.resize(image_permute, dsize=(192, 108), interpolation=cv2.INTER_CUBIC)

        state = image_resize

        self.rew_flag_mem[:] = np.uint8([0]) #reward flag to 0 (false)

        #initialize
        self.licking_cnt = 0
        self.lick_pos.append(self.lick_pos_eps)
        self.lick_pos_eps = []
        self.reward_set.append(self.reward_set_eps)
        self.reward_set_eps = []

        return state
    

    #step to next state
    def step(self, action):
        """
            0: No action
            1: Licking
            2: Move forward
        """
        # Reward
        reward = 0

        #Send action to MATLAB
        if action == 1: #lick
            self.action = np.uint8([1])
            self.action_mem[:] = self.action[:]
            self.action_flag_mem[:] = np.uint8([1])
            self._licking()

            #when reward flag is 1(true, there is reward -> end) & less than 20 licks
            if (self.rew_flag_mem == np.uint8([1])) and (self.licking_cnt <= 20):
                if (self.rew_mem == np.uint8([1])): #reward is 1 (yes reward)
                    reward = self.pos_rew
                else:
                    reward = self.neg_rew
                # reward = self.pos_rew
                # self.rew_flag_mem[:] = np.uint8([0]) #later adding the MATLAB code to make it 0 (false) -> remove this code 
            else: #no reward
                reward = self.neg_rew

        elif action == 2: #move
            self.action = np.uint8([2])
            self.action_mem[:] = self.action[:]
            self.action_flag_mem[:] = np.uint8([1])
            # self._moving()

        #Next state - Get Image from MATLAB
        while (self.img_flag_mem != np.uint8([1])): #if 1(true)
            continue
        image = self.img_mem
        self.img_flag_mem[:] = np.uint8([0])

        image_reshape = np.reshape(image, (3,1920,1080))
        image_permute = image_reshape.transpose((2,1,0))
        image_resize = cv2.resize(image_permute, dsize=(192, 108), interpolation=cv2.INTER_CUBIC)
        next_state = image_resize

        #Done
        done = False
        if (np.array_equal(next_state, self.zeros)): #if black screen, done -> in the agent it uses done to reset?
            done = True

        # Info
        info = {
            "licking_cnt": self.licking_cnt,
            "lick_pos_eps": self.lick_pos_eps
        }

        return next_state, reward, done, info

    def learn(self, state, action, reward, next_state):
        current_q = self.network.forward(torch.tensor(state.copy(), device=self.device).permute(2,0,1).float())[action]
        max_next_q = self.network_target.forward(torch.tensor(next_state.copy(), device=self.device).permute(2,0,1).float()).max()

        expected_q = reward + (self.gamma * max_next_q)
        expected_q = expected_q.reshape([1,1])

        loss = F.mse_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def target_update(self):
        polyak_update(self.network.parameters(), self.network_target.parameters(), self.tau)

    #TRAIN
    def train(self):
        state = self.reset()
        for step in tqdm(range(self.nstep)):
            action = self.act(state)
            next_state, reward, done, _ = self.step(action.item())
            self.learn(state, action, reward, next_state)

            if step % self.FRAME_STEP == 0:
                self.target_update()

            if done:
                self.episode += 1
                print("episode #{} finished\n".format(self.episode))
                next_state = self.reset() 
                while (np.array_equal(next_state, self.zeros)):
                    action = self.act(next_state)
                    next_state, reward, done, _ = self.step(action.item())    

            state = next_state

    def _licking(self):
        self.licking_cnt += 1
        self.lick_pos_eps.append(self.steps_done)

model = VirmenCNN()
print("Train starts...")
model.train()
print("Train ends...")
# save_path = "C:\\Users\\NeuRLab\\RLbench\\data\\ClosedLoop1DTrack_virmen\\a1\\a1s1\\a1s1r1-0\\"
# model.save(os.path.join(save_path, "info.zip"))