""" 
This contains rendering methods for custom algorithms. 
If necessary, copy the load part and apply it to the desired render logic. 
"""

""" IMPORTANT
If a display error appears, run the code below in terminal
(pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None")

BE AWARE THAT this can lead to excessive degradation of rendering time

# Add the gym_duckietown package to your Python path
export PYTHONPATH="${PYTHONPATH}:`pwd`"

# Start xvfb
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &

# Export your display id
export DISPLAY=:1
"""

import os.path as p

import gym
import matplotlib.pyplot as plt
from matplotlib import animation
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import DummyVecEnv, make_atari_env

from custom_envs import OpenLoopStandard1DTrack, OpenLoopTeleportLong1DTrack
from custom_algos import CustomAlgorithm, ALGO_LIST

BASE_PATH = "../data/"
CUSTOM_ENV = {
    "OpenLoopStandard1DTrack": OpenLoopStandard1DTrack,
    "OpenLoopTeleportLong1DTrack": OpenLoopTeleportLong1DTrack
}

def get_atari_env(env_name, n_stack=4):
    env = make_atari_env(env_name, n_envs=1)
    env = VecFrameStack(env, n_stack=n_stack)

    return env

def make_env(env_name):
    if "ALE" in env_name:
        return get_atari_env(env_name)
    elif env_name in CUSTOM_ENV:
        return CUSTOM_ENV[env_name]()
    else:
        return gym.make(env_name)

def get_render_frames(model, env, n_step=10000):
    total_reward = 0
    done_counter = 0
    frames = []

    obs = env.reset()
    for _ in range(n_step):
        # Render into buffer.
        frames.append(env.render(mode="rgb_array"))
        action = model.predict(obs, deterministic=True)[0]

        next_obs, reward, done, _ = env.step(action)
        total_reward += reward[0]
        if done[0]:
            done_counter += 1
            obs = env.reset()
        else:
            obs = next_obs

        if done_counter == 2:
            break
    env.close()
    print(f"Total Reward: {total_reward:.2f}")
    return frames

def display_frames_as_gif(frames, fname="result.gif"):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
        
    ani = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    ani.save(fname, writer='pillow', fps=30)


def main():
    NUM = 1
    ALGO = "DQN"
    ENV = ["ALE/Breakout-v5", "CartPole-v1"][NUM]
    agent_paths = ["a1/a1s1/a1s1r1-0-150003", "a1/a1s1/a1s1r1-0-150012"][NUM]

    file_path = p.join(BASE_PATH, ENV, agent_paths)

    model = CustomAlgorithm.load(
        algo_cls=ALGO_LIST[ALGO],
        path=file_path,
        env=make_env(ENV) # Set this as appropriate code
    )

    frames = get_render_frames(
        model=model,
        env=model.env,
        n_step=500
    )

    display_frames_as_gif(
        frames=frames,
        fname=p.join(file_path, "video.gif")
    )

if __name__ == "__main__":
    main()