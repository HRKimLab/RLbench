from custom_algos.dqn import DQN
from custom_algos.c51 import C51
from custom_algos.qrdqn import QRDQN
from custom_algos.mgdqn import MGDQN
from custom_algos.interface import CustomAlgorithm, init_custom_algo, ALGO_LIST

__all__ = [
    "DQN",
    "C51",
    "QRDQN",
    "MGDQN",
    "CustomAlgorithm",
    "init_custom_algo",
    "ALGO_LIST"
]
