#!/bin/bash

# sb3
conda install pytorch torchvision -c pytorch -y
sudo apt install cmake -y
sudo apt-get install libz-dev -y
pip install pyglet
pip install stable-baselines3[extra]
pip install sb3_contrib

# custom
conda install optuna -c conda-forge
pip install imageio

# render
pip install celluloid
