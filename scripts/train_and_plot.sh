#!/bin/bash

python train.py --env CartPole-v1 --algo custom_dqn --hp default/dqn --nseed 1 --nstep 1000
python plot/plot_mean_combined.py --env CartPole-v1 --agents ['a1s1r1'] --x timesteps --y rew 