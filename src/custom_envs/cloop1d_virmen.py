import pickle

import cv2
import numpy as np
import gym
import imageio
from gym import spaces
from random import randrange
from PIL import Image
import matplotlib.pyplot as plt
from time import time
# from utils.options import get_args


class ClosedLoop1DTrack_virmen(gym.Env):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human', 'gif']}

    def __init__(self, visual_noise=False): #water_spout, video_path,
        # args = get_args()
        # env_list = [[""]]
        self.action_list = [["Stop","Lick","Move","Move+Lick"]]

        #protocol flag - 1: staircase 2: avoidable shock 3: mixed valence
        self.protocol = 3
        if self.protocol == 1:
            num_actions = 4
        elif self.protocol == 2 or self.protocol == 3:
            num_actions = 2

        super().__init__()
        self.action_space = spaces.Discrete(num_actions) # No action / Lick / Move forward / Move+Lick
        
        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(8, 108, 192, 3), dtype=np.uint8
        # ) #changed shape
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(108, 192, 3), dtype=np.uint8
        ) #changed shape

        # self.observation_space = spaces.MultiDiscrete([8, 108, 192, 3]) #changed shape

        # self.water_spout = water_spout
        # self.video_path = video_path
        self.visual_noise = visual_noise #TODO
        
        # self.data = self._load_data()
        self.cur_time = 0
        self.end_time = 300

        #for rendering
        self.cur_pos = 0
        self.start_pos = self.cur_pos
        if self.protocol == 1:
            self.end_pos = 100
        elif self.protocol == 2:
            self.end_pos = 300
        elif self.protocol == 3:
            self.end_pos = 100
        # self.cur_pos = randrange(51, 100) # Remove time-bias
        # self.start_pos = self.cur_pos
        # self.end_pos = randrange(414, 424) # Black screen 
        # self.state = self.data[self.cur_pos, :, :, :]

        self.agent_stat = -1

        self.licking_cnt = 0
        self.licking_after_spout = 0
        self.lick_timing = []
        self.lick_timing_eps = []
        self.spout_timing = []
        self.spout_timing_eps = []
        self.actions_eps = []
        self.actions = []

        #custom part
        #####################################################################################################

        #flag
        self.img_flag = np.uint8([0]) # 0 for false (initialize)
        rew_flag = np.uint8([0])
        self.action_flag = np.uint8([0]) # 1 for true (initialize)
        self.action = np.uint8([0]) #default action

        #file memmap
        image_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_file'
        image_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_flag'
        reward_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_flag'
        reward_mem_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_mem'
        action_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_flag'
        action_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_mem'
        result_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\result_flag'
        ITI_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\ITI_flag'
        step_mem_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\step_mem'
        position_mem_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\position_mem'
        shockzone_start_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\shockzone_start'
        shockzone_end_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\shockzone_end'
        us_type_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\us_type'

        # Memmap shaping
        self.img_mem = np.memmap(image_filename, dtype='uint8',mode='r+', shape=(1080, 1920, 3))
        self.img_flag_mem = np.memmap(image_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.rew_flag_mem = np.memmap(reward_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))   
        self.rew_mem = np.memmap(reward_mem_filename, dtype='uint8',mode='r+', shape=(1, 1))   
        self.action_flag_mem = np.memmap(action_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.action_mem = np.memmap(action_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.result_flag_mem = np.memmap(result_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))  #if the agent reaches the reward zone, it changes to 1(true)
        self.ITI_flag_mem = np.memmap(ITI_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.step_mem = np.memmap(step_mem_filename, dtype='uint16',mode='r+', shape=(1, 1))
        self.position_mem = np.memmap(position_mem_filename, dtype='double',mode='r+', shape=(1, 4))
        self.shockzone_start_mem = np.memmap(shockzone_start_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.shockzone_end_mem = np.memmap(shockzone_end_filename, dtype='uint8',mode='r+', shape=(1, 1))
        self.us_type_mem = np.memmap(us_type_filename, dtype='uint8',mode='r+', shape=(1, 1))

        #initialize
        # self.action_mem[:] = self.action[:] #default
        # self.rew_flag_mem[:]=rew_flag[:]
        # self.img_flag_mem[:] = self.img_flag[:]
        # self.action_flag_mem[:] = self.action_flag[:]
        
        self.data = self.img_mem
        self.state = self.img_mem
        self.previous_state = self.state

        self.ROWS = 108
        self.COLS = 192
        self.FRAME_STEP = 8
        self.EPISODES = 2000
        self.scores, self.episodes, self.average = [], [], []
        self.image_memory = np.zeros((self.FRAME_STEP, self.ROWS, self.COLS, 3), np.uint8)
        self.state_size = (self.FRAME_STEP, self.ROWS, self.COLS, 3)
        
        # # For rendering
        self.frames = []
        # self.original_frames = self._get_original_video_frames() # (682, 1288) #stack the frame one at a time
        self.original_frames = (108,192)
        self.mice_pic = self._load_mice_image()
        self.frame_pos = [[] for i in range(301)]

        # self.zeros = np.zeros(shape = (1080,1920,3), dtype = np.uint8)
        # self.zeros = np.zeros(shape = (108,192,3), dtype = np.uint8)

        #####################################################################################################

        # For plotting
        self.move_timing = []
        self.move_timing_eps = []
        self.lick_pos = []
        self.lick_pos_eps = []
        self.move_and_lick_timing = []
        self.move_and_lick_timing_eps = []
        self.move_and_lick_pos = []
        self.move_and_lick_pos_eps = []
        self.trial_start_timing = []
        self.trial_start_timing_eps = []
        self.shockzone_start_timing_eps = []
        self.shockzone_start_timing = []
        self.shockzone_end_timing_eps = []
        self.shockzone_end_timing = []
        self.shockzone_start_pos_eps = []
        self.shockzone_start_pos = []
        self.shockzone_end_pos_eps = []
        self.shockzone_end_pos = []
        self.shock_timing_eps = []
        self.shock_timing = []
        self.shock_pos_eps = []
        self.shock_pos = []
        self.us_type_eps =[]
        self.us_type = []
        self.time_step_eps = []
        self.time_step = []
        

        # #Temporal variables (for experiments)
        # self.pos_rew = args.pos_rew
        # self.neg_rew = args.lick
        # self.move_neg_rew = args.move
        # self.no_neg_rew = args.stop
        # self.twice_neg_rew = args.move_lick

        #reward initialization
        if self.protocol == 1: #staircase
            self.pos_rew = 1
            self.neg_rew = -0.01
            self.move_neg_rew = -0.01
            self.no_neg_rew = 0
            self.twice_neg_rew = -0.1
        elif self.protocol == 2: #avoidable shock
            self.move_neg_rew = -0.01
            self.no_neg_rew = 0
            self.shock_neg_rew = -10
        elif self.protocol == 3: #mixed valence room
            self.pos_rew = 1
            self.neg_rew = -0.01
            self.no_neg_rew = 0
            self.shock_neg_rew = -10
            self.combined_rew = self.pos_rew + self.shock_neg_rew
       

        self.reward_set = []
        self.reward_set_eps = []

        #for rendering - hyein
        self.q_value_history = [[] for i in range(self.action_space.n)]
        self.y_max = np.NaN
        self.y_min = np.NaN
        self.steps = 0
        self.final_steps = []
        self.td_error = []
        self.time_start = 0

    def get_frame(self):
        # img = state
        # # img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # # print(img_rgb.shape)
        # # img_rgb = np.random.random(size=(400, 600))
        # img_rgb_resized  = cv2.resize(img, (3, self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        # # print(img_rgb_resized.shape)
        # img_rgb_resized[img_rgb_resized < 255] = 0
        # img_rgb_resized = img_rgb_resized / 255
        # # print(img_rgb_resized.shape)

        image = self.img_mem
        self.frame_pos[int(self.position_mem[0][1])] = image
        self.img_flag_mem[:] = np.uint8([0])
        
        image_reshape = np.reshape(image, (3,1920,1080))
        image_permute = image_reshape.transpose((2,1,0))
        image_resize = cv2.resize(image_permute, dsize=(192, 108), interpolation=cv2.INTER_CUBIC)
        # image_resize = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)

        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        # print(self.image_memory)
        self.image_memory[0,:,:,:] = image_resize
        # img = Image.fromarray(self.image_memory[0,:,:,:], 'RGB')
        # img.show()

        # self.imshow(self.image_memory,0) 
        # plt.imshow(self.image_memory[0,:,:])
        # plt.show()
        return np.expand_dims(self.image_memory, axis=0)
    
    def step(self, action):
        """
            0: No action
            1: Licking
            2: Move forward
            3: Move + Lick
        """
        # Execute one time step within the environment
        self.cur_time += 1
        self.actions_eps.append(action)
        self.nstep += 1
        self.steps += 1

        # Record us_type for plotting
        if (self.protocol == 3):
            # self.us_type_eps.append(self.us_type_mem)
            time_elapsed = time()
            if ((time_elapsed-self.time_start >= 2)):
                self.time_step_eps.append(self.cur_time)
                print("==========2 secs last==========")


        # Reward
        reward = 0
        self.n+=1

        if (self.protocol == 3):
            if (self.rew_flag_mem == np.uint8([1]) and self.rew_mem == np.uint8([1])):
                self.us_type_eps.append(1)
                print("water")
            elif (self.rew_flag_mem == np.uint8([1]) and self.rew_mem == np.uint8([2])):
                self.us_type_eps.append(3)
                print("shock")
            elif (self.rew_flag_mem == np.uint8([1]) and self.rew_mem == np.uint8([3])):
                self.us_type_eps.append(4)
                print("combined")

        #send actions to MATLAB
        if action == 0: # no action
            self.action = np.uint8([0])
            self.action_mem[:] = self.action[:]
            self.action_flag_mem[:] = np.uint8([1])
            reward = self.no_neg_rew
            # if self.actions_eps[self.n-1] == self.actions_eps[self.n-2]:
            #     self.count += 1
            # if self.count > 20:
            #     reward = -5
            #     if self.actions_eps[self.n-1] != self.actions_eps[self.n-2]:
            #         self.count = 0
            
        elif (self.protocol == 1 or self.protocol == 3) and action ==1: #lick for staircase & mixed valence
            self.action = np.uint8([1])
            self.action_mem[:] = self.action[:]
            self.action_flag_mem[:] = np.uint8([1])
            self._licking()

            if (self.rew_flag_mem == np.uint8([1]) and self.rew_mem == np.uint8([1])):
                self.licking_after_spout += 1
                if self.licking_after_spout <= 20:   
                    reward = self.pos_rew
                else:
                    reward = self.neg_rew
                self.rew_flag_mem[:] = np.uint8([0])
            elif (self.rew_flag_mem == np.uint8([1]) and self.rew_mem == np.uint8([3])):
                reward = self.combined_rew
                self.rew_flag_mem[:] = np.uint8([0])
                print("shock! {}".format(reward))
                self.shock_timing_eps.append(self.cur_time)
                self.shock_pos_eps.append(self.position_mem[0][1])
            else:
                reward = self.neg_rew

        elif self.protocol == 2 and action == 1: #move for avoidable shock
            self.action = np.uint8([2])
            self.action_mem[:] = self.action[:]
            self.action_flag_mem[:] = np.uint8([1])
            self._moving()
            reward = self.move_neg_rew

        elif action == 2: #move for staircase
            self.action = np.uint8([2])
            self.action_mem[:] = self.action[:]
            self.action_flag_mem[:] = np.uint8([1])
            self._moving()  
            reward = self.move_neg_rew

        elif action == 3: #move+lick for staircase
            self.action = np.uint8([3])
            self.action_mem[:] = self.action[:]
            self.action_flag_mem[:] = np.uint8([1])
            self._moving_and_licking()

            if (self.rew_flag_mem == np.uint8([1]) and (self.rew_mem == np.uint8([1]))):
                self.licking_after_spout += 1
                if self.licking_after_spout <= 20:
                    reward = self.pos_rew
                else:
                    reward = self.twice_neg_rew
                self.rew_flag_mem[:] = np.uint8([0])
            else:
                reward = self.twice_neg_rew

        #get image from MATLAB - wait
        while (self.img_flag_mem != np.uint8([1])):
            continue
        # # print(self.nstep)
        self.step_mem[:] = np.uint16([self.nstep])

        image = self.img_mem
        # self.frame_pos[int(self.position_mem[0][1])] = image
        self.img_flag_mem[:] = np.uint8([0])

        image_reshape = np.reshape(image, (3,1920,1080))
        image_permute = image_reshape.transpose((2,1,0))
        image_resize = cv2.resize(image_permute, dsize=(192, 108), interpolation=cv2.INTER_CUBIC)
        next_state = image_resize
        # img = Image.fromarray(next_state, 'RGB')
        # img.show()

        # next_state = self.get_frame()[0]

        # img = Image.fromarray(next_state[7], 'RGB')
        # img.show()
        # print(next_state)

        # shock
        if (self.protocol == 2):
            if (self.rew_flag_mem == np.uint8([1]) and self.rew_mem == np.uint8([2])):
                reward = self.shock_neg_rew
                self.rew_flag_mem[:] = np.uint8([0])
                print("shock! {}".format(reward))
                self.shock_timing_eps.append(self.cur_time)
                self.shock_pos_eps.append(self.position_mem[0][1])
        elif (self.protocol == 3):
            if (self.rew_flag_mem == np.uint8([1])):
                if (self.rew_mem == np.uint8([2]) or (self.rew_mem == np.uint8([3]) and action != 1)):
                    reward = self.shock_neg_rew
                    self.rew_flag_mem[:] = np.uint8([0])
                    print("shock! {}".format(reward))
                    self.shock_timing_eps.append(self.cur_time)
                    self.shock_pos_eps.append(self.position_mem[0][1])


        if self.shockzone_start_mem == np.uint8([1]):
            self.shockzone_start_timing_eps.append(self.cur_time)
            self.shockzone_start_mem[:] = np.uint8([0])
            self.shockzone_start_pos_eps.append(self.position_mem[0][1])

        if self.shockzone_end_mem == np.uint8([1]):
            self.shockzone_end_timing_eps.append(self.cur_time)
            self.shockzone_end_mem[:] = np.uint8([0])
            self.shockzone_end_pos_eps.append(self.position_mem[0][1])


        print(reward)
       

        # Done
        done = False
        if self.protocol == 1:
            if (self.ITI_flag_mem == np.uint8(1)):
                if (self.pos_rew not in self.reward_set_eps):
                    reward = self.agent_stat
                # else:
                #     self.agent_stat += 0.1
                # if self.agent_stat > 0:
                #     self.agent_stat = 0
                done = True
        
        #done for avoidable shock
        if self.protocol == 2:
            if (self.ITI_flag_mem == np.uint8([1])):
                done = True
                self.ITI_flag_mem[:] = np.uint8([0])
        
        # in protocol 3, not watering in ITI
        if (self.protocol == 3):
            if (self.ITI_flag_mem == np.uint8([1]) and self.rew_flag_mem == np.uint8([1]) and self.rew_mem == np.uint8([1])):
                self.rew_flag_mem[:] = np.uint8([0])
            if (self.ITI_flag_mem == np.uint8([1])):
                done = True
                self.ITI_flag_mem[:] = np.uint8([0])


        # print(self.result_flag_mem)
        if (self.result_flag_mem == np.uint8([1]) and self.rew_mem == np.uint8([1])):
            self.spout_timing_eps.append(self.cur_time)
            # print(self.spout_timing_eps)
            self.spout_pos = self.cur_pos
            # print(self.spout_pos)
            self.result_flag_mem[:] = np.uint8([0]) #result flag to 0(false)

        #trial start timing check for plotting
        if (self.ITI_flag_mem == np.uint8(2)):
            self.trial_start_timing_eps.append(self.cur_time)
            self.time_start = time()
            self.ITI_flag_mem[:] = np.uint8(0)
            print("Trial start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        #to find whether the agent gets shock in moving
        if self.protocol == 2:
            if (action == 1) and (reward <-50):
                f = open("action_reward.txt", "a")
                f.write("{}:{},{}\n".format(self.steps, action,reward))
                f.close()
            
            if (action ==0) and (reward <-50):
                f = open("stop_reward.txt", "a")
                f.write("{}:{},{}\n".format(self.steps, action,reward))
                f.close()

        self.reward_set_eps.append(reward)

        # Info
        info = {
            "cur_time": self.cur_time,
            "start_pos": self.start_pos,
            "cur_pos": self.cur_pos,
            "end_pos": self.end_pos,
            "licking_cnt": self.licking_cnt,
            "lick_pos_eps": self.lick_pos_eps,
            "lick_timing_eps": self.lick_timing_eps,
            "shockzone_start_eps": self.shockzone_start_timing_eps,
            "shockzone_end_eps": self.shockzone_end_timing_eps
            # "episode": self.reward_for_best_model
        }
        # print(reward)

        return next_state, reward, done, info

    def reset(self, stochasticity=True):
        print("reset")
        # Reset the state of the environment to an initial state
        self.cur_time = 0
        self.end_time = 3000

        self.cur_pos = 0
        self.nstep = 0
        self.n = 0
        self.count = 0
        # self.cur_pos = randrange(51, 100) if stochasticity else 51 # Remove time-bias
        # self.start_pos = self.cur_pos
        # self.end_pos = randrange(414, 424) if stochasticity else 414 # Black screen
        # self.state = self.data[self.cur_pos, :, :, :]
        
        while (self.img_flag_mem != np.uint8([1])):
            continue
        self.step_mem[:] = np.uint16([self.nstep])
        # # print(self.nstep)

        image = self.img_mem
        # self.frame_pos[int(self.position_mem[0][1])] = image
        self.img_flag[:] = np.uint8([0]) #make it false (after reading img frame)

        image_reshape = np.reshape(image, (3,1920,1080))
        image_permute = image_reshape.transpose((2,1,0))
        image_resize = cv2.resize(image_permute, dsize=(192, 108), interpolation=cv2.INTER_CUBIC)
        state = image_resize

        # state = self.get_frame()[0]


        self.rew_flag_mem[:] = np.uint8([0]) #reward flag to 0(false)
        # self.result_flag_mem[:] = np.uint8([0]) #result flag to 0(false)
        self.ITI_flag_mem[:] = np.uint8([0]) #ITI flag to 0(false)

        # self.env_mem[:] = self.oloop_standard_env[:] 
        self.licking_cnt = 0
        self.licking_after_spout = 0

        self.move_timing.append(self.move_timing_eps)
        self.move_timing_eps = []
        # print(self.lick_pos_eps)
        self.lick_pos.append(self.lick_pos_eps)
        self.lick_pos_eps = []
        self.reward_set.append(self.reward_set_eps)
        self.reward_set_eps = []
        self.lick_timing.append(self.lick_timing_eps)
        self.lick_timing_eps = []
        self.spout_timing.append(self.spout_timing_eps)
        self.spout_timing_eps = []
        self.actions.append(self.actions_eps)
        self.actions_eps = []
        self.move_and_lick_pos.append(self.move_and_lick_pos_eps)
        self.move_and_lick_pos_eps = []
        self.move_and_lick_timing.append(self.move_and_lick_timing_eps)
        self.move_and_lick_timing_eps = []
        self.trial_start_timing.append(self.trial_start_timing_eps)
        self.trial_start_timing_eps = []
        self.shockzone_start_timing.append(self.shockzone_start_timing_eps)
        self.shockzone_start_timing_eps = []
        self.shockzone_start_pos.append(self.shockzone_start_pos_eps)
        self.shockzone_start_pos_eps = []
        self.shockzone_end_timing.append(self.shockzone_end_timing_eps)
        self.shockzone_end_timing_eps = []
        self.shockzone_end_pos.append(self.shockzone_end_pos_eps)
        self.shockzone_end_pos_eps = []
        self.shock_timing.append(self.shock_timing_eps)
        self.shock_timing_eps = []
        self.shock_pos.append(self.shock_pos_eps)
        self.shock_pos_eps = []
        self.time_step.append(self.time_step_eps)
        self.time_step_eps = []

        if (len(self.us_type_eps) ==0):
            self.us_type_eps.append(0)
        self.us_type.append(self.us_type_eps)
        self.us_type_eps =[]

        return state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        # rgb_array = self.original_frames[self.cur_pos, :, :, :].copy()
        image = self.img_mem[:]
        image_reshape = np.reshape(image, (3,1920,1080))
        image_permute = image_reshape.transpose((2,1,0))
        image_resize = cv2.resize(image_permute, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)
        rgb_array = image_resize
        height, width, _ = rgb_array.shape

        unit_pos = (width - 40) / (self.end_pos - self.start_pos)

        # Upper padding
        padding_height = height // 8
        rgb_array = cv2.copyMakeBorder(rgb_array, padding_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Left padding
        padding_width = width // 10
        rgb_array = cv2.copyMakeBorder(rgb_array, 0, 0, padding_width, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Base line
        base_height = padding_height // 2
        cv2.line(rgb_array, (padding_width + 20, base_height), (padding_width + width - 20, base_height), (0, 0, 0), 2)
        
        # # Current position
        if self.position_mem[0][1] > 300:
            self.position_mem[0][1] = 0
        x_offset = int((self.position_mem[0][1] - self.start_pos) * unit_pos)
        y_offset = base_height - 20
        # print(self.start_pos)
        # print(x_offset, x_offset+self.mice_pic.shape[1])
        # print(self.position_mem[0][1])
        rgb_array[y_offset:y_offset+self.mice_pic.shape[0], padding_width + x_offset:padding_width + x_offset+self.mice_pic.shape[1], :] = self.mice_pic

        if self.protocol == 1:
            # Licking
            for lick_x in self.lick_pos_eps:
                lick_x_pos = padding_width + 20 + int((lick_x - self.start_pos) * unit_pos)
                cv2.line(rgb_array, (lick_x_pos, base_height - 20), (lick_x_pos, base_height + 20), (0, 0, 0), 1)

            # Water spout
            spout_x_pos = 20 + int((96 - self.start_pos) * unit_pos)
            cv2.line(rgb_array, (padding_width + spout_x_pos, base_height - 20), (padding_width + spout_x_pos, base_height + 20), (255, 0, 0), 2)

        if self.protocol == 2:
            try:
                shockzone_start_pos_x = 20 + int((self.shockzone_start_pos_eps[0]-self.start_pos) * unit_pos)
                cv2.line(rgb_array, (padding_width + shockzone_start_pos_x, base_height - 20), (padding_width + shockzone_start_pos_x, base_height + 20), (0, 255, 0), 1)
                shockzone_end_pos_x = 20 + int((self.shockzone_end_pos_eps[0]-self.start_pos) * unit_pos)
                cv2.line(rgb_array, (padding_width + shockzone_end_pos_x, base_height - 20), (padding_width + shockzone_end_pos_x, base_height + 20), (0, 255, 0), 1)
            except:
                print("no shockzone")
            for shock_x in self.shock_pos_eps:
                shock_x_pos = 20 + int((shock_x - self.start_pos) * unit_pos)
                cv2.line(rgb_array, (padding_width + shock_x_pos, base_height - 20), (padding_width + shock_x_pos, base_height + 20), (255, 0, 0), 2)


        if mode == 'human':
            cv2.imshow("licking", rgb_array)
            cv2.waitKey(1)
        elif mode == 'gif':
            self.frames.append(cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB))
        elif mode == 'mp4':
            self.frames.append(rgb_array)
        elif mode == 'rgb_array':
            return rgb_array
            # return cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)

    def save_gif(self):
        imageio.mimsave('video.gif', self.frames, duration=0.005)

    def save_mp4(self, name="test.mp4"):
        height, width, _ = self.frames[0].shape
        fps = 60

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(name, fourcc, float(fps), (width, height))
        for frame in self.frames:
            video.write(frame)
        video.release()

    def _licking(self):
        self.licking_cnt += 1
        self.lick_pos_eps.append(self.position_mem[0][1])
        self.lick_timing_eps.append(self.cur_time)

    def _moving(self):
        self.cur_pos += 1
        self.move_timing_eps.append(self.cur_time)

    def _moving_and_licking(self):
        self.cur_pos += 1
        self.licking_cnt +=1
        self.move_and_lick_pos_eps.append(self.position_mem[0][1])
        self.lick_pos_eps.append(self.position_mem[0][1])
        self.move_and_lick_timing_eps.append(self.cur_time)
        self.lick_timing_eps.append(self.cur_time)

    def _get_original_video_frames(self):
        capture = cv2.VideoCapture(self.video_path)

        frames = []
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frames.append(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        capture.release()
        frames = np.stack(frames, axis=0)

        return frames

    @staticmethod
    def _load_mice_image():
        mice_pic = cv2.imread("../assets/mice.png", cv2.IMREAD_UNCHANGED)
        mice_pic = cv2.resize(mice_pic, (40, 40))
        mice_pic = np.repeat((mice_pic[:, :, 3] < 50).reshape(40, 40, 1), repeats=3, axis=2) * 255
        return mice_pic

    @staticmethod
    def _load_data():
        raise NotImplementedError


class ClosedLoopStandard1DTrack(ClosedLoop1DTrack_virmen):
    """ Licking task in 1D open-loop track with mouse agent """

    metadata = {'render.modes': ['human']}

    def __init__(self, visual_noise=False, pos_rew=10, neg_rew=-5):
        super().__init__(
            water_spout=335,
            video_path="custom_envs/track/VR_standard.mp4",
            visual_noise=visual_noise,
            pos_rew=pos_rew,
            neg_rew=neg_rew
        )

    @staticmethod
    def _load_data():
        with open(f"custom_envs/track/oloop_standard_1d.pkl", "rb") as f:
            data = pickle.load(f)
        return data
