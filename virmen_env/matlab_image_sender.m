%image to array
image = imread('penguin.jpeg'); %example image
shape = size(image); %(131,200,3)

%wait for python's initialization
pause(3);

%file memmap
image_filename = 'C:\\Users\\samsung\\Desktop\\Lab\\RLbench\\virmen_env\\image.dat';
image_flag_filename = 'C:\\Users\\samsung\\Desktop\\Lab\\RLbench\\virmen_env\\image_flag.dat';
action_filename = 'C:\\Users\\samsung\\Desktop\\Lab\\RLbench\\virmen_env\\action.dat';
action_flag_filename = 'C:\\Users\\samsung\\Desktop\\Lab\\RLbench\\virmen_env\\action_flag.dat';

image_mem = memmapfile(image_filename, 'Writable', true, 'Format', {'uint8' [131 200 3] 'image'}); %example image shape
image_flag_mem = memmapfile(image_flag_filename, 'Writable', true, 'Format', {'uint8' [1 1] 'image_flag'}); 
action_mem = memmapfile(action_filename, 'Writable', true, 'Format', {'uint8' [1 1] 'image'});
action_flag_mem = memmapfile(action_flag_filename, 'Writable', true, 'Format', {'uint8' [1 1] 'image_flag'});

while true
    %wait until action_flag is 1(true)
    while (action_flag_mem.data(1).image_flag ~= uint8(1))
        pause(0.25);
    end
    
    %send image
    image_mem.data(1).image(:,:,:) = image(:,:,:);
    
    %get action
    action = action_mem.data(1).action;
    
    %set flag
    image_flag_mem.data(1).image_flag = uint8(1);
    action_flag_mem.data(1).action_flag = uint8(0);
    
end
    