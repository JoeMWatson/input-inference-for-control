import numpy as np
from i2c.exp_types import GaussianI2c, Linearize

ENVIRONMENT = "CartpoleKnown"  # environment to control

# top level training parameters
N_DURATION = 500
N_EPISODE = 1
N_INFERENCE = 200
N_AUG = 1
N_BUFFER = 1
N_STARTING = 0
N_PLOTS = 20
N_ITERS_PER_PLOT = N_INFERENCE / N_PLOTS

POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None

# input inference
linearize = Linearize()
INFERENCE = GaussianI2c(
    inference=linearize,
    Q=np.diag([1.0, 1.0, 100.0, 10.0, 1.0]),
    R=np.diag([1.0]),
    Qf=np.diag([1.0, 1.0, 100.0, 10.0, 1.0]),
    alpha=70.0,
    alpha_update_tol=0.99,
    mu_u=1e-2 * np.random.randn(N_DURATION, 1),
    sig_u=0.25 * np.eye(1),
    mu_x_term=None,
    sig_x_term=None,
)
