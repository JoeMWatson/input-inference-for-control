import numpy as np
from i2c.exp_types import GaussianI2c, Linearize

ENVIRONMENT = "DoubleCartpoleKnown"  # environment to control


# top level training parameters
N_DURATION = 1000
N_EPISODE = 1
N_INFERENCE = 20
N_AUG = 1
N_STARTING = 0
N_PLOTS = 20
N_ITERS_PER_PLOT = N_INFERENCE // N_PLOTS

POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None

# input inference
INFERENCE = GaussianI2c(
    inference=Linearize(),
    Q=np.diag([1.0, 1.0, 100.0, 1.0, 100.0, 1.0, 1.0, 1.0]),
    R=np.diag([0.1]),
    Qf=np.diag([1.0, 1000.0, 1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0]),
    alpha=90.0,
    alpha_update_tol=0.9995,
    mu_u=np.zeros((N_DURATION, 1)),
    sig_u=0.04 * np.eye(1),
    mu_x_term=None,
    sig_x_term=None,
)
