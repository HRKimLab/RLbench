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
│   │   │   ├── data
│   │   │   ├── policy.optimizer.pth
│   │   │   ├── policy.pth
│   │   │   ├── pytorch_variables.pth
│   │   │   ├── _stable_baselines3_version
│   │   │   └── system_info.txt
```

## How to use
```
python plot_numeric --env [ENV_NAME] \
    --agents [AGENT_LIST] \ 
    --x [X-AXIS] \
    --y [Y-AXIS] \
    --data-path [PATH]
```

*example*

```
python plot_numeric.py --env LunarLanderContinuous-v2 \
    --agents "[a1s1r1,a2s1r1,a3s1r1,a4s1r1,a5s1r1,a6s1r1,a7s1r1,a8s1r1]" \
    --x episode \
    --y t \
    --data-path /home/neurlab-dl1/workspace/sb3-practice
```

For more information, please type the following command.  
`python plot_numeric.py --help`
