#I changed Ben's g_agent_testing.m (MATLAB) to Python version
import numpy as np
import time
import random

actions_total = 200
image_size = (270, 480, 3)
run = 1
counter = 0

dummy_actions = [random.randint(0,2) for _ in range(actions_total)]

#file memmap
img_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_file', dtype='uint8',mode='r+', shape=image_size)
img_flag_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_flag', dtype='uint8',mode='r+', shape=(1, 1))
rew_flag_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_flag', dtype='uint8',mode='r+', shape=(1, 1))   
rew_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_mem', dtype='uint8',mode='r+', shape=(1, 1))   
action_flag_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_flag', dtype='uint8',mode='r+', shape=(1, 1))
action_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_mem', dtype='uint8',mode='r+', shape=(1, 1))

start = time.time()
print(start)

while (run == 1):
    if ((action_flag_mem == np.uint8([0])) and (img_flag_mem == np.uint8([1]))):
        if (counter < len(dummy_actions)-1):
            #write action
            action_mem[:] = dummy_actions[counter]

            #access img
            img_mem[:]

            #flip flags back
            img_flag_mem[:] = np.uint8([0])
            action_flag_mem[:] = np.uint8([1])
            # print("run {}".format(counter))
            counter+=1
        else:
            #write action
            action_mem[:] = dummy_actions[counter]

            # #access img
            # img_mem[:]

            #flip flags back
            img_flag_mem[:] = np.uint8([0])
            action_flag_mem[:] = np.uint8([1])
            # print("run {}".format(counter))
            #exit loop
            run = 0


end = time.time()
print(end)
# print("FPS: {}".format(end-start))
print("FPS: {}".format(1/((end-start)/actions_total)))