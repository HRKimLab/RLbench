import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import math
import random

class VirmenCNN:
    def __init__(
        self,
        env = "virmen",
        seed = 0,
        nstep = 1000,
        verbose = 0,
        #hyperparameters
        # EPISODES = 15,
        EPS_START = 0.9,
        EPS_END = 0.05,
        EPS_DECAY = 200
    ):
        self.nstep = nstep
        # self.EPISODES = EPISODES
        self.episode = 0
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0 #학습할 때마다 증가

        self.licking_cnt = 0
        self.lick_pos_eps = []
        self.lick_pos = []
        self.reward_set = []
        self.reward_set_eps = []

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

        self.pos_rew = 10
        self.neg_rew = -5

    #from model find action
    def act(self, state):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            # print(self.network.forward(torch.FloatTensor(obs.copy())).size())
            # torch.LongTensor([[.float()).argmax().item()]], device=self.device)
            net_out = self.network.forward(torch.tensor(obs.copy(), device=self.device).float()).float()
            out = torch.tensor(net_out.argmax().item(), device=self.device).long()
            #return torch.LongTensor([[self.network.forward(torch.FloatTensor(obs.copy()).permute(2,0,1)).argmax().item()]])
            # return torch.LongTensor([[self.network.forward(torch.FloatTensor(obs.copy()).float()).argmax().item()]], device=self.device)
            return out.view(-1, 1)
        else: #무작위value (왼,오) >>> tensor([[0]]) 이런 꼴로 나옴
            return torch.randint(low=0, high=2, size=(1, 1), device=self.device)

    #RESET
    def reset(self):

        #get image
        while (self.img_flag_mem != np.uint8([1])):
            continue
        self.state = self.img_mem
        self.img_flag[:] = np.uint8([0]) #make it false (after reading img frame)

        self.rew_flag_mem[:] = np.uint8([0]) #reward flag to 0 (false)

        #initialize
        self.licking_cnt = 0
        self.lick_pos.append(self.lick_pos_eps)
        self.lick_pos_eps = []
        self.reward_set.append(self.reward_set_eps)
        self.reward_set_eps = []

        return self.state
    

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
        image_reshape = np.reshape(image, (3,1920,1080))
        image_permute = image_reshape.transpose((2,1,0))
        next_state = image_permute

        self.img_flag_mem[:] = np.uint8([0])

        #Done
        done = False
        while (np.array_equal(next_state, self.zeros)): #if black screen, done -> in the agent it uses done to reset?
            done = True
            next_state = self.reset()

        # Info
        info = {
            "licking_cnt": self.licking_cnt,
            "lick_pos_eps": self.lick_pos_eps
        }

        return next_state, reward, done, info

    #TRAIN
    def train(self):
        for step in tqdm(range(self.nstep)):
            action = self.act(state)
            next_state, reward, done, _ = self.step(action.item())
            state = next_state
            if done:
                if self.steps_done % self.FRAME_STEP == 0:
                    self.target_update(100)
                
                self.average.append(sum(self.scores[-100:]) / len(self.scores[-100:]))

                print("episode: {}/{}".format(self.episode, self.EPISODES))

            self.learn()

    def _licking(self):
        self.licking_cnt += 1
        self.lick_pos_eps.append(self.cur_pos)

model = VirmenCNN()
model.train()