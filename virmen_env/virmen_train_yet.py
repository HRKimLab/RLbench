# import gym
# env = gym.make("CartPole-v0")

import numpy as np
from PIL import Image
import time
import pickle
# from stable_baselines3 import DQN
from virmen_dqn import DQN

from utils import (
    set_seed, configure_cudnn, load_json, get_logger,
    get_env, get_algo, set_data_path, clean_data_path, FLAG_FILE_NAME
)


#custom part
##############################################################################

#flag
image_flag = np.uint8([0]) # 0 for false (initialize)
rew_flag = np.uint8([0])
action_flag = np.uint8([1]) # 1 for true (initialize)
action = np.uint8([0]) #default action

#file memmap
# image_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\image.dat'
# image_flag_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\image_flag.dat'
# action_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\action.dat'
# action_flag_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\action_flag.dat'

image_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_file'
image_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_flag'
reward_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_flag'
reward_mem_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_mem'
action_flag_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_flag'
action_filename = 'C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_mem'

# image_mem = np.memmap(image_filename, dtype = 'uint8', mode = 'w+', shape = (131,200,3))
img_mem = np.memmap(image_filename, dtype='uint8',mode='r+', shape=(1080, 1920, 3))
img_flag_mem = np.memmap(image_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))
rew_flag_mem = np.memmap(reward_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))   
rew_mem = np.memmap(reward_mem_filename, dtype='uint8',mode='r+', shape=(1, 1))   
action_flag_mem = np.memmap(action_flag_filename, dtype='uint8',mode='r+', shape=(1, 1))
action_mem = np.memmap(action_filename, dtype='uint8',mode='r+', shape=(1, 1))  

#initialize
action_mem[:] = action[:] #default
img_flag_mem[:] = image_flag[:]
rew_flag_mem[:]=rew_flag[:]
action_flag_mem[:] = action_flag[:]

##############################################################################

env, eval_env = get_env(args.env, args.nenv, save_path, seed)

model = DQN("MlpPolicy", env, img_mem, img_flag_mem, action_mem, action_flag_mem, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()