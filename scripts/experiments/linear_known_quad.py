import numpy as np
from i2c.exp_types import GaussianI2c, CubatureQuadrature

ENVIRONMENT = "LinearKnown"  # environment to control

# top level training parameters
N_DURATION = 60
N_EPISODE = 1
N_INFERENCE = 10
N_AUG = 0
N_STARTING = 0
N_ITERS_PER_PLOT = 1  # N_INFERENCE + 1
POLICY_COVAR = 0 * np.eye(1)
N_PLOTS = 1

# model learning
MODEL = None

# input inference
quad = CubatureQuadrature(1, 0, 0)
INFERENCE = GaussianI2c(
    inference=quad,
    Q=np.diag([10.0, 10.0]),
    R=np.diag([1.0]),
    Qf=np.diag([10.0, 10.0]),
    alpha=800.0,
    alpha_update_tol=0.0,
    mu_u=np.zeros((N_DURATION, 1)),
    sig_u=1.0 * np.eye(1),
    mu_x_term=None,
    sig_x_term=None,
)
