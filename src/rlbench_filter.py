# """ Train agents """

# import os
# from datetime import datetime
# from pathlib import Path

# import torch
# from stable_baselines3.common.noise import (
#     NormalActionNoise,
#     VectorizedActionNoise
# )
# from stable_baselines3.common.logger import configure
# from stable_baselines3.common.callbacks import (
#     EvalCallback, CheckpointCallback
# )

# from utils import (
#     set_seed, configure_cudnn, load_json, get_logger,
#     get_env, get_algo, set_data_path, clean_data_path, FLAG_FILE_NAME
# )
# from utils.options import get_args
# from utils.sb3_callbacks import TqdmCallback, LickingTrackerCallback
# #from stable_baselines.common.callbacks import CheckpointCallback
# from stable_baselines3 import DQN, A2C, PPO

# def train(args):
#     """ Train with multiple random seeds """

#     info_logger, error_logger = get_logger()

#     hp = load_json(args.hp)

#     for i, seed in enumerate(args.seed):
#         set_seed(seed)

#         # Get appropriate path by model info
#         save_path = args.save_path
#         already_run = False
#         if save_path is None:
#             save_path, already_run = set_data_path(args.algo, args.env, hp, seed)

#         # Get env, model
#         try:
#             env, eval_env = get_env(args.env, args.nenv, save_path, seed)
#             action_noise = None
#             if args.noise == "Normal":
#                 assert env.action_space.__dict__.get('n') is None, \
#                     "Cannot apply an action noise to the environment with a discrete action space."
#                 action_noise = NormalActionNoise(args.noise_mean, args.noise_std)
#                 if args.nenv != 1:
#                     action_noise = VectorizedActionNoise(action_noise, args.nenv)
#             model = get_algo(args.algo, env, hp, action_noise, seed)
#         except KeyboardInterrupt:
#             clean_data_path(save_path)
#         except Exception as e:
#             clean_data_path(save_path)
#             info_logger.info("Loading error [ENV: %s] | [ALGO: %s]", args.env, args.algo)
#             error_logger.error("Loading error with [%s / %s] at %s", args.env, args.algo, datetime.now(), exc_info=e)
#             exit()

#         # If given setting had already been run, save_path will be given as None
#         if already_run:
#             print(f"[{i + 1}/{args.nseed}] Already exists: '{save_path}', skip to run")
#         else: # Train with single seed
#             try:
#                 Path(os.path.join(save_path, FLAG_FILE_NAME)).touch()

#                 print(f"[{i + 1}/{args.nseed}] Ready to train {i + 1}th agent - RANDOM SEED: {seed}")
#                 is_licking_task = (args.env in ["OpenLoopStandard1DTrack", "OpenLoopTeleportLong1DTrack"])
#                 _train(
#                     model, args.nstep, is_licking_task,
#                     eval_env, args.eval_freq, args.eval_eps, args.save_freq, save_path
#                 )
#                 del env, model
#             except KeyboardInterrupt:
#                 clean_data_path(save_path)
#             except Exception as e:
#                 clean_data_path(save_path)
#                 info_logger.info("Train error [ENV: %s] | [ALGO: %s]", args.env, args.algo)
#                 error_logger.error("Train error with [%s / %s] at %s", args.env, args.algo, datetime.now(), exc_info=e)


# def _train(
#     model, nstep, is_licking_task,
#     eval_env, eval_freq, eval_eps, save_freq, save_path
# ):
#     """ Train with single seed """

#     # Set logger
#     logger = configure(save_path, ["csv"])
#     model.set_logger(logger)

#     # Set callbacks
#     eval_callback = EvalCallback(
#         eval_env,
#         n_eval_episodes=eval_eps,
#         eval_freq=eval_freq, 
#         log_path=save_path,
#         best_model_save_path=save_path,
#         deterministic=True,
#         verbose=0
#     )
#     tqdm_callback = TqdmCallback()
#     callbacks = [eval_callback, tqdm_callback]
#     if save_freq != -1:
#         checkpoint_callback = CheckpointCallback(
#             save_freq=save_freq,
#             save_path=save_path,
#             name_prefix='rl_model'
#         )
#         callbacks.append(checkpoint_callback)
#     if is_licking_task:
#         licking_tracker_callback = LickingTrackerCallback(
#             env=model.env,
#             save_path=save_path
#         )
#         callbacks.append(licking_tracker_callback)
    

#     # Training
#     model.learn(
#         total_timesteps=nstep,
#         callback=callbacks,
#         eval_log_path=save_path
#     )
    
    
#     os.remove(os.path.join(save_path, FLAG_FILE_NAME))
#     model.save(os.path.join(save_path, "info.zip"))
#     model.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Breakout-v5/a3/a3s1/a3s1r1-0/best_model.zip")


# if __name__ == "__main__":
#     args = get_args()
#     configure_cudnn()

#     print(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} device")
#     print("---START EXPERIMENTS---")
#     train(args)

import torch
from stable_baselines3 import DQN, A2C, PPO
import matplotlib.pyplot as plt
import torch.nn as nn

#model = DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Breakout-v5/a3/a3s1/a3s1r1-0/rl_model_1000000_steps.zip")
model = DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Breakout-v5/a3/a3s1/a3s1r1-0/best_model.zip")
q_values = model.q_net_target
print(q_values)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = q_values.to(device)
#print(model)
model_weights = []
conv_layers = []
list_model_children = []
counter = 0
model_children = list(model.children())
for c in model_children:
    model_children_2 = list(c.children())
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
    
    for weights in model_weights:
        plt.figure(figsize=(20, 17))
        for i, filter in enumerate(weights):
            plt.subplot(8, 8, i+1) 
            plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
            plt.axis('off')
        plt.show()





































# """ Multi-screen rendering & saving for comparing multiple agents """

# import os.path as p
# from celluloid import Camera
# import torch
# import torch.nn as nn

# import matplotlib.pyplot as plt
# from stable_baselines3 import DQN, A2C, PPO
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.env_util import make_atari_env
# from tqdm import tqdm
# from PIL import Image
# from ale_py import ALEInterface

# ale = ALEInterface()


# BASE_PATH = "../data/"

# def get_atari_env(env_name, n_stack=4):
#     env = make_atari_env(env_name, n_envs=1)
#     env = VecFrameStack(env, n_stack=n_stack)

#     return env

# def take_snap(env, ax, name, step=0):
#     ax.imshow(env.render(mode='rgb_array'))
#     ax.text(0.0, 1.01, f"{name} | Step: {step}", transform=ax.transAxes)
#     # ax.set_title(f"{name} | Step: {step}")
#     ax.axis('off')

# def snap_finish(ax, name, step):
#     ax.text(0.0, 1.01, f"{name} | Step: {step}", transform=ax.transAxes)
#     ax.text(
#         .5, .5, 'GAME OVER', 
#         horizontalalignment='center',
#         verticalalignment='center',
#         transform=ax.transAxes
#     )
#     ax.axis('off')


# def render(env_name, models, names, nstep):
#     """ Render how agent interact with environment"""

#     fig_num = len(models)
#     fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
#     plt.subplots_adjust(wspace=0.5)
#     camera = Camera(fig)

#     envs = [get_atari_env(env_name) for _ in range(len(models))]
#     obs = [env.reset() for env in envs]
#     done = [False] * fig_num
#     final_steps = [0] * fig_num

#     delay = 0
#     frames = []
#     for step in tqdm(range(nstep)):
#         for k, (env, name, model) in enumerate(zip(envs, names, models)):
#             print(k)
#             if k % 1000 == 0:
#                 q_values = model.q_net_target
#                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#                 model = q_values.to(device)
#                 #print(model)
#                 model_weights = []
#                 conv_layers = []
#                 list_model_children = []
#                 counter = 0
#                 model_children = list(model.children())
#                 for c in model_children:
#                     model_children_2 = list(c.children())
#                     for i in range(len(model_children_2)):
#                         if type(model_children_2[i]) == nn.Conv2d:
#                             counter+=1
#                             model_weights.append(model_children_2[i].weight)
#                             conv_layers.append(model_children_2[i])
#                         elif type(model_children_2[i]) == torch.nn.modules.container.Sequential:
#                             for j in range(len(model_children_2[i])):
#                                 if type(model_children_2[i][j]) == torch.nn.modules.conv.Conv2d:
#                                     counter+=1
#                                     model_weights.append(model_children_2[i][j].weight)
#                                     conv_layers.append(model_children_2[i][j])
#                     print(f"Total convolution layers: {counter}")
#                     print(conv_layers)
                    
#                     for weights in model_weights:
#                         plt.figure(figsize=(20, 17))
#                         for i, filter in enumerate(weights):
#                             plt.subplot(8, 8, i+1) 
#                             plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
#                             plt.axis('off')
#                         plt.show()
            
#             ax = axs[k // 4][k % 4]
#             #print(model.q_net_target)
#             if not done[k]: 
#                 action, _ = model.predict(obs[k], deterministic=True)
#                 obs[k], _, done[k], info = env.step(action)
                
#                 #q_values = model.q_net_target(torch.tensor(obs[i]).cuda())[0]
#                 #print(q_values)
 

#                 take_snap(env, ax, name, step)
            
#                 if done:
#                     final_steps[k] = step
#             else:
#                 snap_finish(ax, name, final_steps[k])
#         if all(done):
#             delay += 1

#         camera.snap()
#         if delay == 10:
#             break

#     animation = camera.animate()
#     animation.save("animation.mp4", fps=10)


# if __name__ == "__main__":
#     GAME = "ALE/Breakout-v5"
#     agent_paths = [
#         # "a1/a1s1/a1s1r1-0/rl_model_50000_steps",
#         # "a1/a1s1/a1s1r1-0/rl_model_100000_steps",
#         # "a1/a1s1/a1s1r1-0/rl_model_150000_steps",
#         # "a1/a1s1/a1s1r1-0/rl_model_300000_steps",
#         # "a1/a1s1/a1s1r2-42/rl_model_500000_steps",
#         # "a1/a1s1/a1s1r2-42/rl_model_700000_steps",
#         # "a1/a1s1/a1s1r4-7/rl_model_800000_steps",
#         # "a1/a1s1/a1s1r4-7/best_model",
#         "a1/a1s1/a1s1r3-53/best_model"
#     ]

#     # agent_paths = ["a1/a1s1/a1s1r1-0/best_model", "a2/a2s1/a2s1r1-0/best_model"]
#     names = [
#         # "0.05M steps",
#         "0.1M steps"
#         # "0.15M steps",
#         # "0.2M steps",
#         # "0.4M steps",
#         # "0.6M steps",
#         # "0.8M steps",
#         # "1.0M steps",
#     ]

#     agents = [ 
#         DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Breakout-v5/a3/a3s1/a3s1r1-0/best_model.zip")
#         # PPO.load(p.join(BASE_PATH, GAME, agent_path)) for agent_path in agent_paths
#     ]
#     # agents = [
#     #     PPO.load(p.join(BASE_PATH, GAME, agent_paths[0])),
#     #     A2C.load(p.join(BASE_PATH, GAME, agent_paths[1]))
#     # ]
    
#     render(GAME, agents, names, 1000)
