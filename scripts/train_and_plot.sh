#!/bin/bash

python train.py --env CartPole-v1 --algo dqn --hp default/dqn --nseed 3 --nstep 100000
python plot/plot_mean_combined.py --env CartPole-v1 --agents a1s1 --x timesteps --y rew --window_size 100