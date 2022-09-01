

class CustomDQN:
    def __init__(
        self,
        env,
        seed,
        total_timesteps,
        verbose,
        #hyperparameters
        EPISODES,
        EPS_START,
        EPS_END,
        EPS_DECAY = 200,
        GAMMA = 0.8,
        LR = 0.001,
        BATCH_SIZE = 64,

        #from hp
        policy = "MlpPolicy",
        learning_rate = 0.0001,
        buffer_size = 1000000,
        learning_starts = 5000,
        batch_size = 32,
        tau = 1.0,
        gamma = 0.99,
        train_freq = 4,
        gradient_steps = 1,
        target_update_interval = 10000,
        exploration_fraction = 0.1,
        exploration_initial_eps = 1.0,
        exploration_final_eps = 0.05,
        max_grad_norm = 10
    ):
        self.env = 'CartPole-v1'

    #replay buffer
    def memory(self):
        self.memory.append()
