# RLbench

A simple reinforcement learning benchmark framework


## Prerequisites
- Python 3.7+
- PyTorch 1.11.0+
- stable-baselines3 (sb3-contrib) 1.6.0+

## Setup

Tested on Ubuntu 20.04 LTS only.

Create the conda environment, then execute `setup.sh`. This may require a sudo authority. You should type the sudo password during the installation.

```
conda create -n rlbench python=3.9.7
conda activate rlbench

git clone https://github.com/HRKimLab/RLbench.git
cd RLbench/

sh setup.sh
```

**If you want to utilize your GPU when training, please install an appropriate Cuda toolkit that corresponds to your own GPU**

## Quick start
After finishing the [setup](#Setup), change your directory path to `src/` and use the pre-defined script with the following command.
```
sh ../scripts/train_and_plot.sh
```


## Directory structure of data files
### Overall structure 
```
LunarLanderContinuous-v2/
├── a1
│   ├── a1s1
│   │   ├── a1s1r1-7
│   │   ├── a1s1r2-42
│   │   └── a1s1r3-53
│   └── a1s2
│       ├── a1s2r1-7
│       ├── a1s2r2-42
│       └── a1s2r3-53
...

CartPole-v1/
├── a1
│   ├── a1s1
...
```

### Internal files
```
LunarLanderContinuous-v2/
├── a1
│   ├── a1s1
│   │   ├── a1s1r1-0
│   │   │   ├── 0.monitor.csv
│   │   │   ├── best_model.zip
│   │   │   ├── evaluations.npz
│   │   │   ├── info.zip
│   │   │   └── progress.csv
```

## How to use

### Training
At `src/`,

```
python train.py --env [ENV_NAME] \
    --algo [ALGORITHM_NAME] \ 
    --hp [CONFIG_PATH] \
    --nseed [NUMBER_OF_EXPS] \
    --nstep [N_TIMESTEPS] \
    --eval_freq [EVAL_FREQ] \
    --eval_eps [N_EVAL_EPISODES]
```

*example*

```
python train.py --env CartPole-v1 \
    --algo dqn \
    --hp default/dqn \
    --nseed 3 \
    --nstep 100000
```

For more information, please use `--help` option.  
```python train.py --help```


### Train with multiple algorithms and environments
The current implementation only supports running with the same hyperparameters on the multiple experiments

Please modify the hyperparameters in `scripts/run_multiple_trains.py` as you want.  
Then type the following command at the `src/`

```python ../scripts/run_multiple_trains.py```

### Plotting
At `src/`,

```
python plot/plot_mean_combined --env [ENV_NAME] \
    --agents [AGENT_LIST] \ 
    --x [X-AXIS] \
    --y [Y-AXIS]
```

*example*

```
python plot/plot_mean_combined.py --env LunarLanderContinuous-v2 \
    --agents "[a1s1r1,a2s1r1,a3s1r1,a4s1r1,a5s1r1,a6s1r1,a7s1r1,a8s1r1]" \
    --x timesteps \
    --y rew
```

For more information, please use `--help` option.  
```python plot/plot_mean_combined.py --help```
