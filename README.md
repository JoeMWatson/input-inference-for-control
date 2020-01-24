# input-inference-for-control
Input Inference for Control (I2C)
Version 0.0.0 for CoRL 2019 submission

Note: code is still a work in progress. While every effort has been made
to make it clean, there are several prototyped features still under
development in the code (but not used in evaluation).

## Create Environment
```bash
    cd [DIR]
    python3 -m venv env
    source env/bin/activate
    pip3 install -r requirements.txt
```

## Install I2C
```bash
    pip install -e .
```
## Experiments

### LQR Equivalence
```bash
    python scripts/LQR_compare.py
```
results are in _linear/

### Nonlinear Trajectory Optimization

To plot results run
```bash
    python scripts/plot_results.py
```
results are in _plots/

To run controller evalulation see scripts/eval_controller.py
(this requires output from the experiment scripts below)

#### I2C Nonlinear Trajectory Optimization
```bash
    python scripts/run.py pendulum_known
    python scripts/run.py cartpole_known
    python scripts/run.py double_cartpole_known
```
results are in _results/

#### Baseline Nonlinear Trajectory Optimization
```bash
    python scripts/baseline_experiments.py {1}_{2}
```
where 1: {'ilqr', 'gps'}
and 2: {'pendulum', 'cartpole', 'double_cartpole'}

i.e.
```bash
    python scripts/baseline_experiments.py ilqr_pendulum
```
results are in _baselines/

Note: 'trajopt' is Open Source code we have used and edited for the baselines.
We have included it in this repo to preserve anonymity. 
