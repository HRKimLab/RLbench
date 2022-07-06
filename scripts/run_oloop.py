import subprocess

ALGOS = ["dqn", "qrdqn"]
ENVS = ["OpenLoopStandard1DTrack", "OpenLoopTeleportLong1DTrack"]
# ENVS = [env_spec.id for env_spec in envs.registry.all()]

N_SEEDS = 3
N_TIMESTEPS = int(5e4)
EVAL_FREQ = 1000
N_EVAL_EPISODES = 3

for algo in ALGOS:
    for env_id in ENVS:
        args = [
            "--env", env_id,
            "--algo", algo,
            "--hp", f"{algo}_mouse",
            "--nseed", N_SEEDS,
            "--nstep", N_TIMESTEPS,
            "--eval-freq", EVAL_FREQ,
            "--eval-eps", N_EVAL_EPISODES,
        ]
        args = list(map(str, args))
        ok = subprocess.call(["python", "train.py"] + args)
