import subprocess

from gym import envs

ALGOS = ["a2c", "ddpg", "dqn", "ppo", "sac", "td3"]
ENVS = [env_spec.id for env_spec in envs.registry.all()][:600]
# ENVS = [env_spec.id for env_spec in envs.registry.all()]

N_SEEDS = 5
N_TIMESTEPS = int(3e5)
EVAL_FREQ = 5000
N_EVAL_EPISODES = 5

for algo in ALGOS:
    for env_id in ENVS:
        args = [
            "--env", env_id,
            "--algo", algo,
            "--hp", f"default/{algo}",
            "--nseed", N_SEEDS,
            "--nstep", N_TIMESTEPS,
            "--eval-freq", EVAL_FREQ,
            "--eval-eps", N_EVAL_EPISODES,
            "--no-debug"
        ]
        args = list(map(str, args))
        ok = subprocess.call(["python", "train.py"] + args)