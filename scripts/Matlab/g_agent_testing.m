%Acts as an agent without learning, for testing communication speed
%Change 3 variables: g_agent.actions_total and g_agent.image_size
%g_agent.display to set how how many actions are taken, resultion, 
%and displays first 10 pixel values respectively 
%BED - 2022/12/29

function g_agent_testing()

global g_agent
g_agent = []; %initialize g_agent
%preset resolutions
g_agent.image_1080 = [1080 1920 3];
g_agent.image_720 = [720 1280 3];
g_agent.image_480 = [480 640 3];
g_agent.image_270 = [270 480 3];

%%%%%%%%%%%%%%%USER SETTINGS%%%%%%%%%%%%%%%
g_agent.actions_total = 100; %set actions default 200
g_agent.image_size = g_agent.image_1080; %set resolution image_1080 image_720, image_480, image_270
g_agent.display = 0; %0: no display 1: display pixel values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g_agent.run = 1; %initialize while loop
g_agent.run_counter = 1; %counts actions for while loop break

g_agent.dummy_actions = randi([0 2],1,g_agent.actions_total); %random actions between 0, 1, and 2

g_agent.image_out = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\image_file','Writable',true,'Format',{'uint8', g_agent.image_size, 'img'});
g_agent.image_flag = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\image_flag','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_agent.reward_flag = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\reward_flag','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_agent.reward = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\reward_mem','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_agent.action_flag = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\action_flag','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_agent.action = memmapfile('C:\Users\NeuRLab\Documents\MATLAB\action_mem','Writable',true,'Format',{'uint8', [1 1], 'img'});
g_agent.tic = tic();
while g_agent.run == 1
    if g_agent.action_flag.Data.img(1) == 0 && g_agent.image_flag.Data.img(1) == 1
        if g_agent.run_counter < length(g_agent.dummy_actions)
            %write action, move to next action
            g_agent.action.Data.img(1) = g_agent.dummy_actions(g_agent.run_counter);
            g_agent.run_counter = g_agent.run_counter + 1;
            
            %access image
            g_agent.image_out;
            if g_agent.display == 1
                %display image first 10 values
                disp(g_agent.image_out.Data.img(1:10))
            end
            %flip flags back
            g_agent.image_flag.Data.img(1) = 0;
            g_agent.action_flag.Data.img(1) = 1;
        else
            %write final action
            g_agent.action.Data.img(1) = g_agent.dummy_actions(g_agent.run_counter);
                        
            %access image
            g_agent.image_out;
            %flip flags back
            g_agent.image_flag.Data.img(1) = 0;
            g_agent.action_flag.Data.img(1) = 1;
            %exit loop
            g_agent.run = 0;
        end
    end

end
fprintf('actions per second: %.2f in %.0f actions\n',1/(toc(g_agent.tic)/g_agent.actions_total),g_agent.actions_total)
end