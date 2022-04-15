# RLbench

## (Mandatory) Directory structure of data files
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
python plot/plot_numeric --env [ENV_NAME] \
    --agents [AGENT_LIST] \ 
    --x [X-AXIS] \
    --y [Y-AXIS] \
    --data-path [PATH]
```

*example*

```
python plot/plot_numeric.py --env LunarLanderContinuous-v2 \
    --agents "[a1s1r1,a2s1r1,a3s1r1,a4s1r1,a5s1r1,a6s1r1,a7s1r1,a8s1r1]" \
    --x episode \
    --y t \
    --data-path /home/neurlab-dl1/workspace/sb3-practice
```

For more information, please type the following command.  
`python plot/plot_numeric.py --help`
