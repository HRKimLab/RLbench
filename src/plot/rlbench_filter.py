import torch
from stable_baselines3 import DQN, A2C, PPO
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from pathlib import Path
import json
from stable_baselines3 import DQN, A2C, PPO
from filter_options import get_args

def plot_filter(args):

    file_paths = []
    for (root_dir, _, files) in os.walk(os.path.join('/home/neurlab-dl1/workspace/RLbench/data', args.env)):
        for file in files:
            if "rl_model" in file:
                file_path = os.path.join(root_dir, file)
                file_paths.append(file_path)
    sorted_file_paths = sorted(file_paths, key=lambda x: int(x.split('/')[-1].split('_')[-2]))

    for file in sorted_file_paths:
        if args.agent in file:
            path = Path(file)
            json_path = os.path.join(path.parent.parent.parent.absolute(), 'policy.json')
            with open(json_path) as json_file:
                json_data = json.load(json_file)
            algo = json_data["algorithm"]
            dict_algo = {
                'dqn' : DQN,
            }
            model = dict_algo[algo].load(file)
            #NOT SURE
            q_values = model.q_net_target
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = q_values.to(device)
            model_weights = []
            conv_layers = []
            counter = 0
            model_children = list(model.children()) 
            model_children_2 = list(model_children[0].children())
            for i in range(len(model_children_2)):
                if type(model_children_2[i]) == nn.Conv2d:
                    counter+=1
                    model_weights.append(model_children_2[i].weight)
                    conv_layers.append(model_children_2[i])
                elif type(model_children_2[i]) == torch.nn.modules.container.Sequential:
                    for j in range(len(model_children_2[i])):
                        if type(model_children_2[i][j]) == torch.nn.modules.conv.Conv2d:
                            counter+=1
                            model_weights.append(model_children_2[i][j].weight)
                            conv_layers.append(model_children_2[i][j])
            print(f"Total convolution layers: {counter}")
            print(conv_layers)

            if counter == 0:
                print("No convolution layer to visualize")
            else:
                for weights in model_weights:
                    plt.figure(figsize=(20, 17))
                    for i, filter in enumerate(weights):
                        plt.subplot(8, 8, i+1) 
                        plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
                        plt.axis('off')
                    plt.show()

if __name__ == "__main__":
    args = get_args()
    plot_filter(args)
