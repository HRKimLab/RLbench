# RLbench

A simple reinforcement learning benchmark framework (under development)


## Setup

(Tested on Ubuntu 20.04 LTS)  

Create the conda environment, then execute `setup.sh` with sudo (otherwise, you have to type a sudo password during the installation)

For example, 
```
conda create -n rlbench python=3.9.7
conda activate rlbench
sh setup.sh
```

**If you want to utilize your GPU when training, please install an appropriate cuda toolkit which corresponds to your own GPU**


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

### Internal files (expected)
```
LunarLanderContinuous-v2/
├── a1
│   ├── a1s1
│   │   ├── a1s1r1-7
│   │   │   ├── 0.monitor.csv
│   │   │   ├── best_model.zip
│   │   │   ├── evaluations.npz
│   │   │   ├── info.zip
│   │   │   └── progress.csv
```

## How to use
### Training
At `src/` directory,

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
    --algo ppo \
    --hp default/ppo \
    --nseed 3 \
    --nstep 30000
```

For more information, please type the following command.  
`python train.py --help`

### Train with multiple algorithms and environments
**CAUTION** The current implementation only supports running with the same hyperparameters on the multiple experiments

Please modify the hyperparameters in `scripts/run_multiple_trains.py` as you want.  
Then type the following command at the `src/` directory

`python ../scripts/run_multiple_trains.py`

### Plotting
At `src/` directory,

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

For more information, please type the following command.  
`python plot/plot_mean_combined.py --help`
