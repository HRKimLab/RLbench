import torch


class TrainConfig:
    def __init__(
        self,
        run_name: str,
        algo: str,
        env_id: str = "CartPole-v1",
        n_envs: int = 4,
        state_len: int = 1,
        frame_skip: int = 1,
        reward_clipping: bool = False,
        random_seed: int = 42,
        loss_cls: str = "SmoothL1Loss",
        loss_kwargs: dict = {},
        optim_cls: str = "Adam",
        optim_kwargs: dict = {'lr': 0.0003},
        batch_size: int = 128,
        train_step: int = int(1e+6),
        save_freq: int = -1,
        logging_freq: int = 10000,
        device: str = "auto",
        verbose: bool = False
    ):
        assert device in ["cpu", "cuda", "mps", "auto"], "Device should be one of 'cpu', 'cuda', 'mps'."
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is now unavailable."
        elif device == "mps":
            assert torch.backends.mps.is_available(), "mps is now unavailable."
        elif device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.run_name = run_name
        self.algo = algo
        self.env_id = env_id
        self.n_envs = n_envs
        self.state_len = state_len # Sequential images to define state
        self.frame_skip = frame_skip
        self.reward_clipping = reward_clipping
        self.random_seed = random_seed
        self.loss_cls = loss_cls
        self.loss_kwargs = loss_kwargs
        self.optim_cls = optim_cls
        self.optim_kwargs = optim_kwargs
        self.batch_size = batch_size
        self.train_step = train_step
        self.save_freq = save_freq
        self.logging_freq = logging_freq
        self.device = device
        self.verbose = verbose


class DQNConfig:
    def __init__(
        self,
        policy_kwargs: dict,
        eps_cls: str = "LinearDecayLS",
        eps_kwargs: dict = {
            "init_eps": 1.0,
            "milestones": 50000,
            "target_eps": 0.01
        },
        discount_rate: float = 0.98,
        soft_update_rate: float = 1.0,
        buffer_size: int = 100000,
        learning_starts: int = 512,
        train_freq: int = 1,
        target_update_freq: int = 2048,
        n_out: int = -1
    ):
        self.policy_kwargs = policy_kwargs
        self.eps_cls = eps_cls
        self.eps_kwargs = eps_kwargs
        self.discount_rate = discount_rate
        self.soft_update_rate = soft_update_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq


class C51Config(DQNConfig):
    def __init__(
        self,
        policy_kwargs: dict,
        v_min: float = -5.,
        v_max: float = -5.,
        n_atom: int = 51,
        eps_cls: str = "LinearDecayLS",
        eps_kwargs: dict = {
            "init_eps": 1.0,
            "milestones": 50000,
            "target_eps": 0.01
        },
        discount_rate: float = 0.98,
        soft_update_rate: float = 1.0,
        buffer_size: int = 100000,
        learning_starts: int = 512,
        train_freq: int = 1,
        target_update_freq: int = 512,
        n_out: int = -1
    ):
        super().__init__(
            policy_kwargs=policy_kwargs,
            eps_cls=eps_cls,
            eps_kwargs=eps_kwargs,
            discount_rate=discount_rate,
            soft_update_rate=soft_update_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            train_freq=train_freq,
            target_update_freq=target_update_freq
        )
        self.v_min = v_min
        self.v_max = v_max
        self.n_atom = n_atom
        self.n_out = n_atom


class QRConfig(DQNConfig):
    def __init__(
        self,
        policy_kwargs: dict,
        n_quant: int = 50,
        eps_cls: str = "LinearDecayLS",
        eps_kwargs: dict = {
            "init_eps": 1.0,
            "milestones": 50000,
            "target_eps": 0.01
        },
        discount_rate: float = 0.98,
        soft_update_rate: float = 1.0,
        buffer_size: int = 100000,
        learning_starts: int = 512,
        train_freq: int = 1,
        target_update_freq: int = 512,
        n_out: int = -1
    ):
        super().__init__(
            policy_kwargs=policy_kwargs,
            eps_cls=eps_cls,
            eps_kwargs=eps_kwargs,
            discount_rate=discount_rate,
            soft_update_rate=soft_update_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            train_freq=train_freq,
            target_update_freq=target_update_freq
        )
        self.n_quant = n_quant
        self.n_out = n_quant


class MGDQNConfig:
    def __init__(
        self,
        policy_kwargs: dict,
        gamma_min: float = 0.8,
        gamma_max: float = 0.99,
        gamma_n: int = 10,
        soft_vote: bool = False,
        eps_cls: str = "LinearDecayLS",
        eps_kwargs: dict = {
            "init_eps": 1.0,
            "milestones": 50000,
            "target_eps": 0.01
        },
        soft_update_rate: float = 1.0,
        buffer_size: int = 100000,
        learning_starts: int = 512,
        train_freq: int = 1,
        target_update_freq: int = 2048,
        n_out: int = -1
    ):
        self.policy_kwargs = policy_kwargs
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_n = gamma_n
        self.soft_vote = soft_vote
        self.n_out = gamma_n
        self.eps_cls = eps_cls
        self.eps_kwargs = eps_kwargs
        self.soft_update_rate = soft_update_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq


ALGO_CONFIG = {
    'DQN': DQNConfig, 'C51': C51Config, 'QRDQN': QRConfig, 'MGDQN': MGDQNConfig
}


__all__ = [
    "ALGO_CONFIG",
    "TrainConfig",
    "DQNConfig",
    "C51Config",
    "QRConfig",
    "MGDQNConfig",
]