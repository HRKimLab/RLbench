function g_agent_testing()

global g_agent

g_agent = [];
g_agent.actions_total = 200;
% g_agent.image_size = [1080 1920 3];
g_agent.image_size = [720 1280 3];
g_agent.run = 1;
g_agent.run_counter = 1;

g_agent.dummy_actions = randi([0 2],1,g_agent.actions_total);

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
disp(1/(toc(g_agent.tic)/g_agent.actions_total))
end