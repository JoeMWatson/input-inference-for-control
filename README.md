# input-inference-for-control
Approximate Inference for Stochastic Optimal Control

[comment]: <> (![Trajectory Optimization]&#40;assets/dcp_10s.gif&#41;)

[comment]: <> (![Covariance Control]&#40;assets/p_cc_10s.gif&#41;)
<img src="https://github.com/JoeMWatson/input-inference-for-control/blob/master/assets/dcp_10s.gif" width="400"/>
<img src="https://github.com/JoeMWatson/input-inference-for-control/blob/master/assets/p_cc_10s.gif" width="400"/>

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/1910.03003)
[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2103.06319)
[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2105.07693)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)

[comment]: <> ([![Pytorch 1.3]&#40;https://img.shields.io/badge/pytorch-1.3.1-blue.svg&#41;]&#40;https://pytorch.org/&#41;)

[comment]: <> ([![License: MIT]&#40;https://img.shields.io/badge/License-MIT-yellow.svg&#41;]&#40;https://github.com/cambridge-mlg/arch_uncert/blob/master/LICENSE&#41;)

What is it?
-----------
Input Inference for Control (`i2c`) is an inference-based optimal control algorithm. 
The current implementation, Gaussian `i2c`, can perform trajectory optimization, model predictive control and covariance control via a Gaussian approximation of the optimal state-action distribution. This yields time-varying linear (Gaussian) controllers, and is approximately equivalent to quadratic optimal control methods like differential dynamic programming, iterative-/sequential LQR.

For more information, see the following papers:

[1] J. Watson and H. Abdulsamad and R. Findeisen and J. Peters. *Stochastic Control through Approximate Bayesian Input Inference.* Submitted to IEEE Transactions on Automatic Control Special Issue, Learning and Control 2021. ([arXiv](https://arxiv.org/abs/2105.07693))

[2] J. Watson and J. Peters. *Advancing Trajectory Optimization with Approximate Inference: Exploration, Covariance Control and Adaptive Risk.* American Control Conference (ACC) 2021 ([arXiv](https://arxiv.org/abs/2103.06319))

[3] J. Watson and H. Abdulsamad and J. Peters. *Stochastic Optimal Control as Approximate Input Inference.* Conference on Robot Learning (CoRL) 2019. ([arXiv](https://arxiv.org/abs/1910.03003))

Installation
-----------
Create environment `i2c` and install
```bash
    cd input-inference-for-control && conda create -y -n i2c pip python=3.7 && conda activate i2c && pip3 install -r requirements.txt && pip install -e .
```

Example
--------
To optimize pendulum swing-up with cubature quadrature, run
```bash
    python scripts/i2c_run.py pendulum_known_quad
```
the output directory results should look like this

<img src="https://github.com/JoeMWatson/input-inference-for-control/blob/master/assets/pendulum_msg_iter_0_199.png" width="600"/>


Experiments
-----------
Prior experiments are preserved here. All results are stored in `/_results`.

### LQR Equivalence
Section 3.1 of [3]
```bash
    python scripts/LQR_compare.py
```

### Nonlinear Trajectory Optimization
Section 3.2 of [3], Section IV.A of [1]
```bash
    python scripts/i2c_run.py -h
```
results are in _results/


### Linear Gaussian Covariance Control
Linear Gaussian covariance control.
Section IV.C of [2]
```bash
    python scripts/linear_covariance_control.py
```

### Nonlinear Gaussian Covariance Control
Pendulum swing-up with covaraince control.
Section IV.C of [2]
```bash
    python scripts/nonlinear_covariance_control.py
```

### Model Predictive Control with State Estimation
Runs `i2c` and iLQR MPC with a cubature kalman filter for an acrobatic quadropter task.
Section IV.C of [1]
```bash
    python scripts/mpc_state_est/mpc_quad.py 0 --plot
```

Baselines
---------
For iLQR use https://github.com/hanyas/trajopt 

Citing Input Inference for Control
----------------------------------
To cite `i2c`, please reference the appropriate paper
```BibTeX
@misc{watson2021stochastic,
      title={Stochastic Control through Approximate Bayesian Input Inference}, 
      author={Watson, Joe and Abdulsamad, Hany and Findeisen, Rolf and Peters, Jan},
      year={2021},
      eprint={2105.07693},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```BibTeX
@inproceedings{i2cacc,
	author    = {Watson, Joe and Peters, Jan},
	title     = {Advancing Trajectory Optimization with Approximate Inference: Exploration, Covariance Control and Adaptive Risk},
	booktitle = {American Control Conference},
	year      = {2021},
}
```
```BibTeX
@inproceedings{i2ccorl,
	author    = {Watson, Joe and  Abdulsamad, Hany and Peters, Jan},
	title     = {Stochastic Optimal Control as Approximate Input Inference},
	booktitle = {Conference on Robot Learning},
	year      = {2019},
}
```
