import subprocess

ALGO = "dqn"
ENV = "OpenLoopStandard1DTrack"
# ENVS = [env_spec.id for env_spec in envs.registry.all()]

N_SEEDS = 2
N_TIMESTEPS = int(5e4)
EVAL_FREQ = 1000
N_EVAL_EPISODES = 3

pos_rews = [x for x in range (200, 1000, 100)]
neg_rews = [x for x in range(-15, 0, 5)] + [x for x in range(-100, 0, 20)]

for pos_rew in pos_rews:
    for neg_rew in neg_rews:
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
