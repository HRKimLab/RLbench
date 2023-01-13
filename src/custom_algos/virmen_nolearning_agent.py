import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import time

class VirmenCNN:
    def __init__(
        self,
        env = "virmen",
        seed = 0,
        nstep = 200,
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

        # image_size = (1080, 1920, 3)
        image_size = (270, 480, 3)
        # image_size = (270, 480, 3)

        self.img_mem = np.memmap(image_filename, dtype='uint8',mode='r+', shape=image_size)
        self.img_flag_mem = np.memmap(image_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.rew_flag_mem = np.memmap(reward_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))   
        self.rew_mem = np.memmap(reward_mem_filename, dtype='uint8',mode='r+', shape=(1, 1))   
        self.action_flag_mem = np.memmap(action_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.action_mem = np.memmap(action_filename, dtype='uint8',mode='r+', shape=(1, 1))

        #initialize
        # self.action_mem[:] = self.action[:] #default
        # self.rew_flag_mem[:]=self.rew_flag[:]
        # self.img_flag_mem[:] = self.img_flag[:]
        # self.action_flag_mem[:] = self.action_flag[:]

        #black image
        # self.zeros = np.zeros(shape = (108,192,3), dtype = np.uint8)
        self.zeros = np.zeros(shape = image_size, dtype = np.uint8)

        self.pos_rew = 10
        self.neg_rew = -5

    #only action 2 : move
    def act(self, state):
        return np.uint8(2)

    #RESET
    def reset(self):
        #get image
        while (self.img_flag_mem != np.uint8([1])):
            continue
        state_image = self.img_mem
        self.img_flag[:] = np.uint8([0]) #make it false (after reading img frame)

        # image_reshape = np.reshape(state_image, (3,1920,1080))
        # image_permute = image_reshape.transpose((2,1,0))
        # image_resize = cv2.resize(image_permute, dsize=(192, 108), interpolation=cv2.INTER_CUBIC)

        image_reshape = np.reshape(state_image, (3,480,270))
        image_resize = image_reshape.transpose((2,1,0))
        # image_resize = cv2.resize(image_permute, dsize=(192, 108), interpolation=cv2.INTER_CUBIC)

        state = image_resize

        #see the reshaped image
        # image = Image.fromarray(image_resize, 'RGB')
        # image.show()

        #initialize
        self.licking_cnt = 0
        self.lick_pos.append(self.lick_pos_eps)
        self.lick_pos_eps = []
        self.reward_set.append(self.reward_set_eps)
        self.reward_set_eps = []

        self.rew_flag_mem[:] = np.uint8([0]) #reward flag to 0 (false)

        return state
    

    #step to next state
    def step(self, action):
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

        # image_reshape = np.reshape(image, (3,1920,1080))
        # image_permute = image_reshape.transpose((2,1,0))
        # image_resize = cv2.resize(image_permute, dsize=(192, 108), interpolation=cv2.INTER_CUBIC)
        image_reshape = np.reshape(image, (3,480,270))
        image_resize = image_reshape.transpose((2,1,0))
        # image_resize = cv2.resize(image_permute, dsize=(192, 108), interpolation=cv2.INTER_CUBIC)
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

    #TRAIN
    def train(self):
        state = self.reset()
        for step in tqdm(range(self.nstep)):
            action = self.act(state)
            next_state, reward, done, _ = self.step(action.item())

            if done:
                self.episode += 1
                # print("episode #{} finished".format(self.episode))
                next_state = self.reset() 
                # while (np.array_equal(next_state, self.zeros)):
                #     action = self.act(next_state)
                #     next_state, reward, done, _ = self.step(action.item())    

            state = next_state

    def _licking(self):
        self.licking_cnt += 1
        self.lick_pos_eps.append(self.steps_done)

model = VirmenCNN()
print("Train starts...")
start = time.time()
model.train()
print("Train ends...")
end = time.time()
print("FPS {}".format(1/((end-start)/200)))