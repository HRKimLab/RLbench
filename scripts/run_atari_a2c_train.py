import subprocess

from gym import envs

ALGO = "a2c"
ENVS = [
    "ALE/VideoPinball-v5", "ALE/Boxing-v5", "ALE/Breakout-v5", "ALE/StarGunner-v5", 
    "ALE/Robotank-v5", "ALE/Atlantis-v5", "ALE/CrazyClimber-v5", "ALE/Gopher-v5",
    "ALE/DemonAttack-v5", "ALE/NameThisGame-v5", "ALE/Krull-v5", "ALE/Assault-v5",
    "ALE/RoadRunner-v5", "ALE/Kangaroo-v5", "ALE/Jamesbond-v5", "ALE/Tennis-v5",
    "ALE/Pong-v5", "ALE/SpaceInvaders-v5", "ALE/BeamRider-v5", "ALE/Tutankham-v5"
    "ALE/KungFuMaster-v5", "ALE/Freeway-v5", "ALE/TimePilot-v5", "ALE/Enduro-v5",
    "ALE/FishingDerby-v5", "ALE/UpNDown-v5", "ALE/IceHockey-v5", "ALE/Qbert-v5",
    "ALE/Hero-v5", "ALE/Asterix-v5", "ALE/BattleZone-v5", "ALE/WizardOfWor-v5",
    "ALE/ChopperCommand-v5", "ALE/Centipede-v5", "ALE/BankHeist-v5", "ALE/Riverraid-v5",
    "ALE/Zaxxon-v5", "ALE/Amidar-v5", "ALE/Alien-v5", "ALE/Venture-v5",
    "ALE/Seaquest-v5", "ALE/DoubleDunk-v5", "ALE/Bowling-v5", "ALE/MsPacman-v5",
    "ALE/Asteroids-v5", "ALE/Frostbite-v5", "ALE/Gravitar-v5", "ALE/PrivateEye-v5", 
    "ALE/MontezumaRevenge-v5"
]

N_SEEDS = 3
N_TIMESTEPS = int(3e6)
EVAL_FREQ = 100000
N_EVAL_EPISODES = 10

for env_id in ENVS:
    args = [
        "--env", env_id,
        "--algo", ALGO,
        "--hp", f"{ALGO}_atari",
        "--nseed", N_SEEDS,
        "--nstep", N_TIMESTEPS,
        "--eval-freq", EVAL_FREQ,
        "--eval-eps", N_EVAL_EPISODES,
        "--no-debug"
    ]
    args = list(map(str, args))
    ok = subprocess.call(["python", "train.py"] + args)