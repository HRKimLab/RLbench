import numpy as np

#file memmap
img_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_file', dtype='uint8',mode='r+', shape=(1080, 1920, 3))
img_flag_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_flag', dtype='uint8',mode='r+', shape=(1, 1))
rew_flag_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_flag', dtype='uint8',mode='r+', shape=(1, 1))   
rew_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_mem', dtype='uint8',mode='r+', shape=(1, 1))   
action_flag_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_flag', dtype='uint8',mode='r+', shape=(1, 1))
action_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_mem', dtype='uint8',mode='r+', shape=(1, 1))

