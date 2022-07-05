""" Rendering """
import time

from stable_baselines3 import DQN
from custom_envs import OpenLoopStandard1DTrack

model_path = "../data/OpenLoopStandard1DTrack/a1/a1s1/a1s1r2-42/best_model.zip"
env = OpenLoopStandard1DTrack()

model = DQN.load(model_path)

for _ in range(1):
    total_reward = 0
    state = env.reset()
    while True:
        action, _ = model.predict(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        time.sleep(.005)
        env.render()
    print(f"Total reward: {total_reward}")