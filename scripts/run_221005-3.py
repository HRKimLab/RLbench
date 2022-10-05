import subprocess

ALGO = "dqn"
ENV = ["ClosedLoopStandard1DTrack"]
# ENVS = [env_spec.id for env_spec in envs.registry.all()]

N_SEEDS = 2
N_TIMESTEPS = int(2e5)
EVAL_FREQ = 5000
N_EVAL_EPISODES = 3

reward_sets = [
    (10, -5),
    (100, -10),
    (1000, -10),
    (5, -100),
    (50, -100),
    (100, -100),
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
