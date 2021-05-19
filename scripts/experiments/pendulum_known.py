import numpy as np
from i2c.exp_types import GaussianI2c, Linearize

ENVIRONMENT = "PendulumKnown"  # environment to control

# top level training parameters
N_DURATION = 100
N_EPISODE = 1
N_INFERENCE = 150
N_AUG = 0
N_STARTING = 0
N_PLOTS = 1
N_ITERS_PER_PLOT = N_INFERENCE / N_PLOTS
N_BUFFER = 1

POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None

# input inference
linearize = Linearize()
INFERENCE = GaussianI2c(
    inference=linearize,
    Q=np.diag([1, 100.0, 1]),
    R=np.diag([1]),
    Qf=np.diag([1, 100.0, 1]),
    alpha=100.0,
    alpha_update_tol=0.99,
    mu_u=np.zeros((N_DURATION, 1)),
    sig_u=0.2 * np.eye(1),
    mu_x_term=None,
    sig_x_term=None,
)
