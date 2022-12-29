%Acts as virmen without running it, for testing communication speed
%Change 2 variables: g_virmen.frame_get and g_virmen.image_size 
%to set how often it refreshes the frame and resultion respectively 
%BED - 2022/12/29

function dummy_virmen()
global g_virmen
g_virmen = [];

g_virmen.frame_get = 0; %0 = refresh frame once, 1 = refresh frame every action

g_virmen.image_720 = [720 1280 3];
g_virmen.image_480 = [480 640 3];
g_virmen.image_270 = [270 480 3];

g_virmen.image_size = g_virmen.image_720; %set resolution

g_virmen.getframe_size = [0 0 g_virmen.image_size(2) g_virmen.image_size(1)];
g_virmen.figsize = zeros(g_virmen.image_size(1),g_virmen.image_size(2));

g_virmen.run = 1; %initialize while loop
imshow(g_virmen.figsize); %make plot for getframe

if exist('C:\Users\NeuRLab\Documents\MATLAB\image_file','file')
   delete('C:\Users\NeuRLab\Documents\MATLAB\image_file');
   if exist('C:\Users\NeuRLab\Documents\MATLAB\image_file','file')
       error('cannot delete file ..  image_file');
   end
end
fid = fopen('C:\Users\NeuRLab\Documents\MATLAB\image_file', 'w');
fwrite(fid, zeros(g_virmen.image_size), 'uint8'); %initialization
fclose(fid);

g_virmen.image_out = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\image_file','Writable',true,'Format',{'uint8', g_virmen.image_size, 'img'});
g_virmen.image_flag = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\image_flag','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.reward_flag = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\reward_flag','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.reward = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\reward_mem','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.action_flag = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\action_flag','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.action = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\action_mem','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.action_flag.Data.img(1) = 1; 
g_virmen.image_flag.Data.img(1) = 0;

if g_virmen.frame_get == 0
    g_virmen.frame=getframe(1,g_virmen.getframe_size); %get new frame
end

while g_virmen.run == 1;
    if g_virmen.action_flag.Data.img(1) == 1 && g_virmen.image_flag.Data.img(1) == 0
        g_virmen.action_process = g_virmen.action.Data.img(1); %intake action
        
        if g_virmen.frame_get == 1
            g_virmen.frame=getframe(1,g_virmen.getframe_size); %get new frame
        end
        
        g_virmen.image_out = g_virmen.frame.cdata;
        
        g_virmen.action_flag.Data.img(1) = 0;
        g_virmen.image_flag.Data.img(1) = 1;
    end
end

end