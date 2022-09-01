from math import gamma
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy
import logging
import random

class CustomDQN:
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.001,#1e-4,
        total_timesteps: int = 1000,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 64,
        tau: float = 1.0,
        gamma: float = 0.8,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
    ):
        # super().__init__(
        #     policy,
        #     env,
        #     learning_rate,
        #     buffer_size,
        #     learning_starts,
        #     batch_size,
        #     tau,
        #     gamma,
        #     train_freq,
        #     gradient_steps,
        #     action_noise=None,  # No action noise
        #     replay_buffer_class=None,
        #     replay_buffer_kwargs=None,
        #     policy_kwargs=policy_kwargs,
        #     tensorboard_log=None,
        #     verbose=verbose,
        #     device=device,
        #     create_eval_env=False,
        #     seed=seed,
        #     sde_support=False,
        #     optimize_memory_usage=False,
        #     supported_action_spaces=(gym.spaces.Discrete,),
        #     support_multi_env=True,
        # )

        self.env = gym.make('CartPole-v1')
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.tau = tau
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm

        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None
        self._setup_model()
        self.optimizer = optim.Adam(self.q_net.parameters(), learning_rate)

    def _setup_model(self) -> None:
        # super()._setup_model()
        # self.q_net = self.policy.q_net
        # self.q_net_target = self.policy.q_net_target
        self.q_net = nn.Sequential(
            #input layer=4, hidden layer=256, output layer=2
            #input layer =[]
            nn.Linear(4,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )
        # self.q_net_target = nn.Sequential(
        #     #input layer=4, hidden layer=256, output layer=2
        #     #input layer =[]
        #     nn.Linear(4,120),
        #     nn.ReLU(),
        #     nn.Linear(120,84),
        #     nn.ReLU(),
        #     nn.Linear(84,2)
        # )
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def memorize(self, obs, action, reward, next_obs):
        self.memory.append((
            torch.FloatTensor([obs]),
            torch.LongTensor([[action]]),
            torch.FloatTensor([reward]),
            torch.FloatTensor([next_obs])
        ))

    def _on_step(self) -> None: # Not implemented!
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        # self.logger = logging.getLogger()
        ###### 매 스텝마다 실행되는 함수 ######
        ###### HINT: exploration rate을 사용하는 에이전트일 경우, self.exploration_schedule을 활용할 것 ######
        ###### 예) new_exp_rate = self.exploration_schedule(old_exp_rate) -> exploration rate이 업데이트됨.

        # logger = logging.getLogger

        self._n_calls += 1
        # if self._n_calls % self.target_update_interval == 0 :
        #     polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        #update exploration rate
        current_progress_remaining = 1 - (self._n_calls / self.total_timesteps)
        self.exploration_rate = self.exploration_schedule(current_progress_remaining)
        # self.exploration_rate = 0.2
        #print(self.exploration_rate)
        # self.exploration_rate = get_linear_fn(self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction)
        # self.logger.record("rollout/exploration_rate", self.exploration_rate)



        # logger.record("exploration_rate", self.exploration_rate)
        #update target network
        # q_net_target_weight = []
        # q_net_weight = []

        # a = list(list(self.q_net_target.children())[1].children())
        # for layer in a:
        #     if type(layer) == torch.nn.modules.linear.Linear:
        #         q_net_target_weight.append(layer.weight)

        # b = list(list(self.q_net.children())[1].children())
        # for layer in b:
        #     if type(layer) == torch.nn.modules.linear.Linear:
        #         q_net_weight.append(layer.weight)
        
        # if self._n_calls % self.target_update_interval == 0 :
        #     for i in range(len(q_net_target_weight)):
        #         q_net_target_weight[i] = self.tau * q_net_weight[i] + (1 - self.tau) * q_net_target_weight[i]
        # if self._n_calls % self.target_update_interval == 0 :
        #     polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
 



    def train(self, gradient_steps: int, obs: np.ndarray,  deterministic: bool) -> None: # Not implemented!
        # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        #self._update_learning_rate(self.policy.optimizer)

        t_step = 0
        for _ in range(gradient_steps):
            
            if self._n_calls <= self.total_timesteps:
                action, obs = self.predict(observation = obs, deterministic = deterministic)
                # action = np.array(action)
                # action = int(np.random(1) >= 0.5)
                next_obs, reward, done, _ = self.env.step(action)
                losses = []

                self.env.render()

                if done:
                    reward = -1


                ###### replay buffer를 사용하는 알고리즘일 경우, 위해 아래 코드를 활용할 것
                # Sample replay buffer
                # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

                self.memorize(obs, action, reward, next_obs)

                if done:
                    obs = self.env.reset()
                    print(f"reset! t_step = {t_step}")
                    t_step=0
                else:
                    obs = next_obs
                    t_step += 1       

                self._on_step()


                if len(self.memory) >= self.batch_size:
                    #경험이 충분히 쌓일 때부터 학습 진행
                    batch = random.sample(self.memory, self.batch_size)
                    obss, actions, rewards, next_obss = zip(*batch)
            
                    obss = torch.cat(obss)
                    actions = torch.cat(actions)
                    rewards = torch.cat(rewards)
                    next_obss = torch.cat(next_obss)

                    with torch.no_grad():
                        
                        current_q = self.q_net(obss).gather(1,actions).reshape(-1)

                        # max_next_q = self.q_net_target(next_obss).detach().max(1)[0].reshape(-1)
                        max_next_q = self.q_net(next_obss).detach().max(1)[0].reshape(-1)
                        expected_q = rewards + (self.gamma* max_next_q)
                        
                        #current_q = self.q_net(obs).detach().max(0)[0]

                        # max_next_q = self.q_net_target(next_obs).detach().max(0)[0]
                        # expected_q = reward + (1 - done) * (self.gamma * max_next_q)


                        loss = F.mse_loss(current_q.squeeze(), expected_q)
                        loss.requires_grad_(True)
                        losses.append(loss.item())
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                else:
                    pass
                
                    
                ###### Not implemented!



                # if done:
                #     obs = torch.FloatTensor(self.env.reset())
                #     print(f"reset! t_step = {t_step}")
                #     t_step=0
                # else:
                #     obs = next_obs
                #     t_step += 1
                
                # Increase update counter
                # self._n_updates += gradient_steps
                # self._on_step()


            else:
                break

    def predict( # Not implemented!
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None, # Do not erase this
        deterministic: bool = True # Do not erase this
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.randn() < self.exploration_rate:
            action = self.env.action_space.sample()
        else:
            action = self.q_net(torch.FloatTensor(observation)).detach().max(0)[1].item() # >> tensor([0])
            #action = self.q_net(observation).detach().argmax()  # >> tensor(0)
        # action = self.env.action_space.sample()

        # action, state = np.array([0]), None
        # Not implemented!
        return action, observation
        # return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:
        print('learn')
        #return super().learn(
        #     total_timesteps=total_timesteps,
        #     callback=callback,
        #     log_interval=log_interval,
        #     eval_env=eval_env,
        #     eval_freq=eval_freq,
        #     n_eval_episodes=n_eval_episodes,
        #     tb_log_name=tb_log_name,
        #     eval_log_path=eval_log_path,
        #     reset_num_timesteps=reset_num_timesteps,
        # )
        obs = self.env.reset()
        # self.train(self, gradient_steps = 1000, obs = obs, deterministic = False, batch_size = 100)
        self.train(100000, obs, False)
        # self.train(self, gradient_steps = 1000, batch_size = 100)
    def set_logger(error_log, info_log):
        return 
    
    def save(a, b):
        print('2')

    def _excluded_save_params(self) -> List[str]:
        print('_excluded_save_params')
        # return super()._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []


# agent = CustomDQN(DQNPolicy, "CartPole-v1")
# agent._setup_model()
# agent.learn(total_timesteps=1000)