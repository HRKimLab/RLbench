%Acts as virmen without running it, for testing communication speed
%Change 2 variables: g_virmen.frame_get and g_virmen.image_size 
%to set how often it refreshes the frame and resultion respectively 
%BED - 2022/12/29

function dummy_virmen()

global g_virmen
g_virmen = [];
% g_virmen.image_size = [720 1280 3];
% g_virmen.getframe_size = [0 0 1280 720];
% g_virmen.figsize = zeros(1280,720);
% g_virmen.image_size = [270 480 3];
% g_virmen.getframe_size = [0 0 480 270];
% g_virmen.figsize = zeros(480,270);
g_virmen.image_size = [108 192 3];
g_virmen.getframe_size = [0 0 192 108];
g_virmen.figsize = zeros(192,108);
g_virmen.run = 1;
imshow(g_virmen.figsize);

g_virmen.image_out = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\image_file','Writable',true,'Format',{'uint8', g_virmen.image_size, 'img'});
g_virmen.image_flag = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\image_flag','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.reward_flag = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\reward_flag','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.reward = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\reward_mem','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.action_flag = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\action_flag','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.action = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\action_mem','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_virmen.action_flag.Data.img(1) = 1; 
g_virmen.image_flag.Data.img(1) = 0;

g_virmen.frame=getframe(1,g_virmen.getframe_size); %get new frame

while g_virmen.run == 1;
    
    if g_virmen.action_flag.Data.img(1) == 1 && g_virmen.image_flag.Data.img(1) == 0
        g_virmen.action_process = g_virmen.action.Data.img(1); %intake action
        
        if g_virmen.frame_get == 1
            g_virmen.frame.cdata = uint8(200*rand(size(g_virmen.frame.cdata)));
        end
        
        g_virmen.image_out.Data.img = g_virmen.frame.cdata;
        
        g_virmen.action_flag.Data.img(1) = 0;
        g_virmen.image_flag.Data.img(1) = 1;
    end
end

end