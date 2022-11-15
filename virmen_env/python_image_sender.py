import numpy as np
from PIL import Image
import time
import pickle

#image to array
image = Image.open('penguin.jpeg') #example image
image_array = np.asarray(image)
shape = image_array.shape #(131,200,3)

# #array to image
# image = Image.fromarray(image_array, 'RGB')
# image.show()

# #get image - reshape, permute
# image_reshape = np.reshape(image_mem, (3,200,131))
# image_permute = image_reshape.transpose((2,1,0))

#flag
image_flag = np.uint8([0]) # 0 for false (initialize)
action_flag = np.uint8([1]) # 1 for true (initialize)
action = np.uint8([0]) #default action

#file memmap
image_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\image.dat'
image_flag_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\image_flag.dat'
action_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\action.dat'
action_flag_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\action_flag.dat'

# image_mem = np.memmap(image_filename, dtype = 'uint8', mode = 'w+', shape = (131,200,3))
image_mem = np.memmap(image_filename, dtype = 'uint8', mode = 'w+', shape = (160,210,3))
image_flag_mem = np.memmap(image_flag_filename, dtype = 'uint8', mode = 'w+', shape = (1,1))
action_mem = np.memmap(action_filename, dtype = 'uint8', mode = 'w+', shape = (1,1))
action_flag_mem = np.memmap(action_flag_filename, dtype = 'uint8', mode = 'w+', shape = (1,1))

#initialize
action_mem[:] = action[:] #default
image_flag_mem[:] = image_flag[:]
action_flag_mem[:] = action_flag[:]

#image pickle file
with open('C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\src\\custom_envs\\track\\oloop_standard_1d.pkl', 'rb') as f:
    oloop_standard_env = pickle.load(f)
ind = 0

while True:
    #wait until image_flag is 1(true)
    while (action_flag_mem != np.uint8([1])):
        time.sleep(0.1)
    
    #get action
    action = action_mem[:]

    if (action == np.uint8([2])):
        ind += 1

    #send image
    image_mem[:] = oloop_standard_env[ind]  

    #set flag
    image_flag_mem[:] = np.uint8([1])
    action_flag_mem[:] = np.uint8([0])
