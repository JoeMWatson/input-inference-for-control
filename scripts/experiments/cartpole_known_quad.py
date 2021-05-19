import numpy as np
from i2c.exp_types import GaussianI2c, CubatureQuadrature

ENVIRONMENT = "CartpoleKnown"  # environment to control

# top level training parameters
N_DURATION = 500
N_EPISODE = 1
N_INFERENCE = 100
N_AUG = 1
N_BUFFER = 1
N_STARTING = 0
N_PLOTS = 8
N_ITERS_PER_PLOT = N_INFERENCE / N_PLOTS

POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None


# input inference
INFERENCE = GaussianI2c(
    inference=CubatureQuadrature(1, 0, 0),
    Q=np.diag([1.0, 1.0, 100.0, 10.0, 1.0]),
    R=np.diag([1.0]),
    Qf=np.diag([1.0, 1.0, 100.0, 10.0, 1.0]),
    alpha=80.0,
    alpha_update_tol=0.0,
    mu_u=1e-3 * np.random.randn(N_DURATION, 1),
    sig_u=1.0 * np.eye(1),  # was 1.0
    mu_x_term=None,
    sig_x_term=None,
)
