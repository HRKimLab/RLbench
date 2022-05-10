import gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

from options import get_args

def render(env, model, nstep):
    """ Render how agent interact with environment"""

    env = gym.make("CartPole-v1")
    obs = env.reset() # 환경 초기화
    # obs : observation ( = state)

    ## 학습 시작
    for i in range(nstep): # nstep : 5000
        action, _state = model.predict(obs, deterministic=True) # 0, 1
        obs, reward, done, info = env.step(action)

        # env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    # trained_model = DQN.load("/home/neurlab-dl1/workspace/RLbench/data/ALE/Breakout-v5/a1/a1s1/a1s1r2-42/best_model.zip", verbose=1)
    # env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=0)
    # env = VecFrameStack(env, n_stack=4)
    # render(env, trained_model, 25_000)

    trained_model = DQN.load("/home/neurlab-dl1/workspace/RLbench/data/CartPole-v1/a1/a1s1/a1s1r2-42/best_model.zip")
    env = gym.make("CartPole-v1")
    render(env, trained_model, 25_000)