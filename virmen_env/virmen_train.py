# import gym
# env = gym.make("CartPole-v0")

import numpy as np
from PIL import Image
import time
import pickle
# from stable_baselines3 import DQN
from agent_dqn import DQN

from utils import (
    set_seed, configure_cudnn, load_json, get_logger,
    get_env, get_algo, set_data_path, clean_data_path, FLAG_FILE_NAME
)


#custom part
##############################################################################

#flag
image_flag = np.uint8([0]) # 0 for false (initialize)
action_flag = np.uint8([1]) # 1 for true (initialize)
action = np.uint8([0]) #default action

#file memmap
image_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\image.dat'
image_flag_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\image_flag.dat'
action_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\action.dat'
action_flag_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\action_flag.dat'

env_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\env.dat'

# image_mem = np.memmap(image_filename, dtype = 'uint8', mode = 'w+', shape = (131,200,3))
image_mem = np.memmap(image_filename, dtype = 'uint8', mode = 'w+', shape = (160,210,3))
image_flag_mem = np.memmap(image_flag_filename, dtype = 'uint8', mode = 'w+', shape = (1,1))
action_mem = np.memmap(action_filename, dtype = 'uint8', mode = 'w+', shape = (1,1))
action_flag_mem = np.memmap(action_flag_filename, dtype = 'uint8', mode = 'w+', shape = (1,1))

env_mem = np.memmap(env_filename, dtype = 'uint8', mode = 'w+', shape = (459,160,210,3))

#initialize
action_mem[:] = action[:] #default
image_flag_mem[:] = image_flag[:]
action_flag_mem[:] = action_flag[:]

#send env
with open('C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\src\\custom_envs\\track\\oloop_standard_1d.pkl', 'rb') as f:
    oloop_standard_env = pickle.load(f)

env_mem[:] = oloop_standard_env[:]

##############################################################################

env, eval_env = get_env(args.env, args.nenv, save_path, seed)

model = DQN("MlpPolicy", env, image_mem, image_flag_mem, action_mem, action_flag_mem, verbose=1)
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