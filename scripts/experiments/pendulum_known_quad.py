import numpy as np
from i2c.exp_types import GaussianI2c, CubatureQuadrature

ENVIRONMENT = "PendulumKnown"  # environment to control

# top level training parameters
N_DURATION = 100
N_EPISODE = 1
N_INFERENCE = 200
N_AUG = 0
N_STARTING = 0
N_PLOTS = 1
N_ITERS_PER_PLOT = N_INFERENCE / N_PLOTS
N_BUFFER = 0

POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None

# input inference
INFERENCE = GaussianI2c(
    inference=CubatureQuadrature(1, 0, 0),
    Q=np.diag([1, 100.0, 1]),
    R=np.diag([2]),
    Qf=np.diag([1, 100.0, 1]),
    alpha=100,
    alpha_update_tol=0.0,
    mu_u=1e-2 * np.random.randn(N_DURATION, 1),
    sig_u=2.0 * np.eye(1),
    mu_x_term=None,
    sig_x_term=None,
)
