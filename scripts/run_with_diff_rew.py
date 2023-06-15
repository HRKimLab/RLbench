import subprocess

from gym import envs

ALGO = "dqn"
HP = "dqn_mouse"
ENV = "ClosedLoop1DTrack_virmen"
N_SEEDS = 3
N_TIMESTEPS = 6000
EVAL_FREQ = 5000
N_EVAL_EPISODES = 1
POS_REW = 5
MOVE_LICK_REW = [-1, -1]
MOVE_REW = 0
LICK_REW = 0
STOP_REW = 0
SAVE_FREQ = 500

for rew in MOVE_LICK_REW:
    args = [
        "--env", ENV,
        "--algo", ALGO,
        "--hp", "dqn_mouse",
        "--nseed", N_SEEDS,
        "--nstep", N_TIMESTEPS,
        "--pos_rew", POS_REW,
        "--move_lick", rew,
        "--move", MOVE_REW,
        "--lick", LICK_REW,
        "--stop", STOP_REW,
        "--eval-freq", EVAL_FREQ,
        "--eval-eps", N_EVAL_EPISODES,
        "--save-freq", SAVE_FREQ
    ]
    args = list(map(str, args))
    ok = subprocess.call(["python", "..\\src\\train.py"] + args)