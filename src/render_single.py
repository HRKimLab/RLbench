""" Rendering """
import time

from stable_baselines3 import DQN
from custom_envs import OpenLoopStandard1DTrack

model_path = "../data/Leg_OpenLoopStandard1DTrack/a1/a1s1/a1s1r1-0/best_model.zip"
env = OpenLoopStandard1DTrack()

model = DQN.load(model_path)
state = env.reset()

for _ in range(3):
    state = env.reset()
    while True:
        action, _ = model.predict(state)
        state, reward, done, _ = env.step(action)
        if done:
            break

        time.sleep(.005)
        env.render()
