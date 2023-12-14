from typing import Union, Iterable, Tuple
from collections.abc import Iterable as Iterable_
from math import prod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MlpPolicy(PolicyNetwork):
    def __init__(
        self,
        algo: str,
        n_actions: int,
        input_size: int,
        hidden_sizes: Iterable[int] = [64, ],
        state_len: int = 1,
        n_out: Union[int, Iterable[int]] = -1
    ):
        """
            n_out: If given, network will be built for C51 algorithm
        """

        super().__init__()
        self.algo = algo
        self.in_layer = nn.Linear(input_size * state_len, hidden_sizes[0])
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_shape, out_shape) for in_shape, out_shape in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        if n_out == -1:
            self.fc_q = nn.Linear(hidden_sizes[-1], n_actions)
        else:
            self.fc_q = nn.Linear(
                hidden_sizes[-1],
                n_actions * (n_out if isinstance(n_out, int) else prod(n_out))
            )

        self.n_actions = n_actions
        self.n_out = n_out
        self.state_len = state_len

        self._init_weight()
    
    def forward(self, x):
        if self.state_len != 1:
            x = x.flatten(-2)
        x = F.relu(self.in_layer(x))
        for layer in self.linears:
            x = F.relu(layer(x))

        out = self.fc_q(x) # (B, n_act * X)
        if isinstance(self.n_out, int) and self.n_out != -1:
            out = out.view(-1, self.n_actions, self.n_out)
        elif isinstance(self.n_out, Iterable_):
            out = out.view(-1, self.n_actions, *self.n_out)
        
        if self.algo == "C51":
            out = F.softmax(out, dim=-1)

        return out.squeeze() # squeeze required for the case when n_env == 1


class CnnPolicy(PolicyNetwork):
    def __init__(
        self,
        algo: str,
        n_actions: int,
        state_len: int = 1,
        n_out: Union[int, Iterable[int]] = -1,
        input_shape: Tuple[int] = (84, 84, 3),
        rgb_array: bool = False
    ):
        """
            n_out: If given, network will be built for C51 algorithm
        """

        super().__init__()
        self.algo = algo

        # Expected input tensor shape: (B, state_len, 84, 84)
        # Input (B, 210, 160, 3) will be processed by `ProcessFrame84` wrapper -> (B, 84, 84, state_len)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        
        example_input = torch.randn((input_shape[2], input_shape[0], input_shape[1]))
        self.fc = nn.Linear(self.conv(example_input).flatten().shape[0], 512)

        if n_out == -1:
            self.fc_q = nn.Linear(512, n_actions)
        else:
            self.fc_q = nn.Linear(
                512,
                n_actions * (n_out if isinstance(n_out, int) else prod(n_out))
            )

        # action value distribution
        self.n_actions = n_actions
        self.n_out = n_out
        self.state_len = state_len
        self.rgb_array = rgb_array

        self._init_weight()

    def forward(self, x):
        if x.dim() == 2: # When n_envs == 1 and state_len == 1 and not rgb_array
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3 and (self.state_len == 1 and not self.rgb_array):
            x = x.unsqueeze(1)
        elif x.dim() == 3 and (self.state_len != 1 or self.rgb_array):
            x = x.unsqueeze(0)

        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x / 255.0)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))

        out = self.fc_q(x) # (B, n_act * X)
        if isinstance(self.n_out, int) and self.n_out != -1:
            out = out.view(-1, self.n_actions, self.n_out)
        elif isinstance(self.n_out, Iterable_):
            out = out.view(-1, self.n_actions, *self.n_out)

        if self.algo == "C51":
            out = F.softmax(out, dim=-1)

        return out.squeeze() # squeeze required for the case when n_env == 1


def get_policy_networks(
    algo: str,
    policy_type: str,
    state_len: int,
    n_act: Union[int, Iterable[int]],
    n_in: Union[int, Iterable[int]],
    n_out: Union[int, Iterable[int]] = -1,
    input_shape=(84, 84, 3),
    hidden_sizes: Iterable[int] = [64, ],
    rgb_array: bool = False
) -> PolicyNetwork:
    if policy_type == "MlpPolicy":
        return MlpPolicy(
            algo=algo,
            n_actions=n_act,
            input_size=n_in,
            hidden_sizes=hidden_sizes,
            state_len=state_len,
            n_out=n_out
        )
    elif policy_type == "CnnPolicy":
        return CnnPolicy(
            algo=algo,
            n_actions=n_act,
            state_len=state_len,
            n_out=n_out,
            input_shape=input_shape,
            rgb_array=rgb_array
        )
    else:
        raise ValueError(policy_type)


if __name__ == "__main__":
    mlp_test_arguments = (
        ["DQN", 2, 4, [64, 64], 1, -1],
        ["C51", 2, 4, [64, ], 1, 51],
        ["QRDQN", 2, 4, [64, ], 1, 20],
        ["MGDQN", 2, 4, [64, 64], 1, 10],
        ["MGC51", 2, 4, [64, ], 1, (10, 51)],
    )
    cnn_test_arguments = (
        ["DQN", 2, 1, -1],
        ["C51", 2, 1, 51],
        ["QRDQN", 2, 1, 20],
        ["MGDQN", 2, 1, 10],
        ["MGC51", 2, 1, (10, 51)],
    )
    
    BATCH_SIZE = 4

    print("Policy: MlpPlicy")
    for n_env in (1, 4):
        for state_len in (1, 4):
            for i, args in enumerate(mlp_test_arguments):
                print(f"Algorithm: {args[0]} | n_env: {n_env} | state_len: {state_len}")
                model = MlpPolicy(*args)
                tr_input = torch.randn(BATCH_SIZE, args[4], 4)
                if n_env == 1:
                    inf_input = torch.randn(args[4], 4)
                else:
                    inf_input = torch.randn(n_env, args[4], 4)
                
                if args[0] == "DQN":
                    training_answer = torch.randn(BATCH_SIZE, args[1])
                elif args[0] == "C51":
                    training_answer = torch.randn(BATCH_SIZE, args[1], args[-1])
                elif args[0] == "QRDQN":
                    training_answer = torch.randn(BATCH_SIZE, args[1], args[-1])
                elif args[0] == "MGDQN":
                    training_answer = torch.randn(BATCH_SIZE, args[1], args[-1])
                elif args[0] == "MGC51":
                    training_answer = torch.randn(BATCH_SIZE, args[1], *args[-1])

                if args[0] == "DQN":
                    inference_answer = torch.randn(n_env, args[1])
                elif args[0] == "C51":
                    inference_answer = torch.randn(n_env, args[1], args[-1])
                elif args[0] == "QRDQN":
                    inference_answer = torch.randn(n_env, args[1], args[-1])
                elif args[0] == "MGDQN":
                    inference_answer = torch.randn(n_env, args[1], args[-1])
                elif args[0] == "MGC51":
                    inference_answer = torch.randn(n_env, args[1], *args[-1])
                
                if n_env == 1:
                    inference_answer = inference_answer.squeeze(0)

                assert model(tr_input).shape == training_answer.shape,\
                    f"Incorrect! Training stage | Input: {tr_input.shape} | Output: {model(tr_input).shape}"
                assert model(inf_input).shape == inference_answer.shape,\
                    f"Incorrect! Inference stage | Input: {inf_input.shape} | Output: {model(inf_input).shape}"
                print("OK!")
            print("----------------------------")


    print("Policy: CnnPlicy")
    for n_env in (1, 4):
        for state_len in (1, 4):
            for i, args in enumerate(cnn_test_arguments):
                print(f"Algorithm: {args[0]} | n_env: {n_env} | state_len: {state_len}")
                model = CnnPolicy(*args)
                tr_input = torch.randn(BATCH_SIZE, args[2], 84, 84)
                if n_env == 1:
                    inf_input = torch.randn(args[2], 84, 84)
                else:
                    inf_input = torch.randn(n_env, args[2], 84, 84)
                
                if args[0] == "DQN":
                    training_answer = torch.randn(BATCH_SIZE, args[1])
                elif args[0] == "C51":
                    training_answer = torch.randn(BATCH_SIZE, args[1], args[-1])
                elif args[0] == "QRDQN":
                    training_answer = torch.randn(BATCH_SIZE, args[1], args[-1])
                elif args[0] == "MGDQN":
                    training_answer = torch.randn(BATCH_SIZE, args[1], args[-1])
                elif args[0] == "MGC51":
                    training_answer = torch.randn(BATCH_SIZE, args[1], *args[-1])

                if args[0] == "DQN":
                    inference_answer = torch.randn(n_env, args[1])
                elif args[0] == "C51":
                    inference_answer = torch.randn(n_env, args[1], args[-1])
                elif args[0] == "QRDQN":
                    inference_answer = torch.randn(n_env, args[1], args[-1])
                elif args[0] == "MGDQN":
                    inference_answer = torch.randn(n_env, args[1], args[-1])
                elif args[0] == "MGC51":
                    inference_answer = torch.randn(n_env, args[1], *args[-1])
                
                if n_env == 1:
                    inference_answer = inference_answer.squeeze(0)

                assert model(tr_input).shape == training_answer.shape,\
                    f"Incorrect! Training stage | Input: {tr_input.shape} | Output: {model(tr_input).shape}"
                assert model(inf_input).shape == inference_answer.shape,\
                    f"Incorrect! Inference stage | Input: {inf_input.shape} | Output: {model(inf_input).shape}"
                print("OK!")
            print("----------------------------")
