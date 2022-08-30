import subprocess

ALGO = "dqn"
ENV = "OpenLoopStandard1DTrack"
# ENVS = [env_spec.id for env_spec in envs.registry.all()]

N_SEEDS = 3
N_TIMESTEPS = int(5e4)
EVAL_FREQ = 1000
N_EVAL_EPISODES = 3

for pos_rew in range(0, 100, 5):
    for neg_rew in range(-100, 0, 5):
        args = [
            "--env", ENV,
            "--algo", ALGO,
            "--hp", f"{ALGO}_mouse",
            "--nseed", N_SEEDS,
            "--nstep", N_TIMESTEPS,
            "--eval-freq", EVAL_FREQ,
            "--eval-eps", N_EVAL_EPISODES,
            "--pos-rew", pos_rew,
            "--neg-rew", neg_rew
        ]
        args = list(map(str, args))
        ok = subprocess.call(["python", "train.py"] + args)
