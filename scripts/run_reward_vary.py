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

reward_sets = [
    (5, -100),
    (10, -100),
    (20, -100),
    (50, -100),
    (100, -100),

    (100, -5),
    (100, -10),
    (300, -5),
    (300, -10),
    (500, -5),
    (500, -10),
    (1000, -5),
    (1000, -10),
    (5000, -5),
    (5000, -10),
]
for (pos_rew, neg_rew) in reward_sets:
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
