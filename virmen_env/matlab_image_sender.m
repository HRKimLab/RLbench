%image to array
image = imread('penguin.jpeg'); %example image
shape = size(image); %(131,200,3)

%wait for python's initialization
pause(3);

%file memmap
image_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\image.dat';
image_flag_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\image_flag.dat';
action_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\action.dat';
action_flag_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\action_flag.dat';

env_filename = 'C:\\Users\\NeuRLab\\Desktop\\Lab\\RLbench\\virmen_env\\env.dat'

image_mem = memmapfile(image_filename, 'Writable', true, 'Format', {'uint8' [160 210 3] 'image'}); %example image shape
image_flag_mem = memmapfile(image_flag_filename, 'Writable', true, 'Format', {'uint8' [1 1] 'image_flag'}); 
action_mem = memmapfile(action_filename, 'Writable', true, 'Format', {'uint8' [1 1] 'action'});
action_flag_mem = memmapfile(action_flag_filename, 'Writable', true, 'Format', {'uint8' [1 1] 'action_flag'});

env_mem = memmapfile(env_filename, 'Writable', true, 'Format', {'uint8' [459 160 210 3] 'env'});

ind = 1;
oloop_standard_env = env_mem.data(1).env;


while true
    %wait until action_flag is 1(true)
    while (action_flag_mem.data(1).action_flag ~= uint8(1))
        pause(0.25);
    end
    
    %get action
    action = action_mem.data(1).action;
    
    %if (action == uint8(1))
    %    image(50:100,:,:) = uint8(0);
    %end
    
    if (action == uint8(2))
        ind = ind +1;
    end
    
    %send image
    %image_mem.data(1).image(:,:,:) = image(:,:,:);
    env = squeeze(oloop_standard_env(ind,:,:,:));
    reshape = reshape(env, [3 210 160]);
    oloop_permute = permute(reshape, [3,2,1]);
    image_mem.data(1).image(:,:,:) = squeeze(oloop_permute(ind,:,:,:));
    
    %set flag
    image_flag_mem.data(1).image_flag = uint8(1);
    action_flag_mem.data(1).action_flag = uint8(0);
    
end
    