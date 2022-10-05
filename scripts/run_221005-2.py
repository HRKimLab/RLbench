import subprocess

ALGO = "dqn"
ENV = ["InterleavedOpenLoop1DTrack"]
# ENVS = [env_spec.id for env_spec in envs.registry.all()]

N_SEEDS = 2
N_TIMESTEPS = int(2e5)
EVAL_FREQ = 5000
N_EVAL_EPISODES = 3

reward_sets = [
    (12, -5),
    (101, -10),
    (1001, -10),
    (6, -100),
    (51, -100),
    (101, -100),
]

for env in ENV:
    for (pos_rew, neg_rew) in reward_sets:
        args = [
            "--env", env,
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
