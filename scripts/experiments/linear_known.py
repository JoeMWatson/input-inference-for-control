import numpy as np
from i2c.exp_types import GaussianI2c, Linearize

ENVIRONMENT = "LinearKnown"  # environment to control

# top level training parameters
N_DURATION = 60
N_EPISODE = 1
N_INFERENCE = 30
N_AUG = 0
N_STARTING = 0
N_PLOTS = 1
N_ITERS_PER_PLOT = 1  # N_INFERENCE + 1
POLICY_COVAR = 0 * np.eye(1)

# model learning
MODEL = None

# input inference
INFERENCE = GaussianI2c(
    inference=Linearize(),
    Q=np.diag([10.0, 10.0]),
    R=np.diag([1.0]),
    Qf=np.diag([10.0, 10.0]),
    alpha=1e2,
    alpha_update_tol=0.0,
    mu_u=np.zeros((N_DURATION, 1)),
    sig_u=1e2 * np.eye(1),
    mu_x_term=None,
    sig_x_term=None,
)
