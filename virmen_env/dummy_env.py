import numpy as np
from PIL import Image
import time
import pickle

#image to array
# image = np.random.rand(500, 270, 480, 3)

start = time.time()
# image = np.zeros((500, 270, 480, 3))
# shape = image.shape 
print(time.time() - start)
#flag
image_flag = np.uint8([0]) # 0 for false (initialize)
action_flag = np.uint8([1]) # 1 for true (initialize)
action = np.uint8([2]) #default action

#file memmap
img_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_file', dtype='uint8',mode='r+', shape=(720, 1280, 3))
img_flag_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\image_flag', dtype='uint8',mode='r+', shape=(1, 1))
rew_flag_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_flag', dtype='uint8',mode='r+', shape=(1, 1))   
rew_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\reward_mem', dtype='uint8',mode='r+', shape=(1, 1))   
action_flag_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_flag', dtype='uint8',mode='r+', shape=(1, 1))
action_mem = np.memmap('C:\\Users\\NeuRLab\\Documents\\MATLAB\\action_mem', dtype='uint8',mode='r+', shape=(1, 1))

#initialize
action_mem[:] = action[:] #default
img_flag_mem[:] = image_flag[:]
action_flag_mem[:] = action_flag[:]

ind = 0
run = 1

# while True:
#     #wait until image_flag is 1(true)
#     # while (action_flag_mem != np.uint8([1])):
#     #     time.sleep(0.001)
    
#     #get action
#     action = action_mem[:]

#     if (action == np.uint8([2])):
#         ind += 1

#     #send image
#     img_mem[:] = image[ind]  

#     #set flag
#     img_flag_mem[:] = np.uint8([1])
#     action_flag_mem[:] = np.uint8([0])
print(time.time() - start)
while (run == 1):
    start = time.time()
    if ((action_flag_mem == np.uint8([1])) and (img_flag_mem == np.uint8([0]))):
        action = action_mem[:]
        print('action mem: {}\n'.format(time.time() - start))
        start = time.time()
        # if action == np.uint8([2]):
        #     ind += 1

        print('action if: {}\n'.format(time.time() - start))
        ind+=1
        print(ind)
        if ind < 5:
            img = np.ones((720,1280,3))
        else:
            img = np.zeros((720, 1280, 3))
        img_mem[:] = img

        img_flag_mem[:] = np.uint8([1])
        action_flag_mem[:] = np.uint8([0])
        print('others: {}\n'.format(time.time() - start))
        print(img_flag_mem, action_flag_mem, action_mem)
    # print(time.time() - start)
print(time.time() - start)