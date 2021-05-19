import numpy as np
from i2c.exp_types import GaussianI2c, Linearize

ENVIRONMENT = "LinearKnownMinimumEnergy"  # environment to control

# top level training parameters
N_DURATION = 50
N_EPISODE = 1
N_INFERENCE = 15
# N_INFERENCE = 500
N_AUG = 0
N_STARTING = 0
N_PLOTS = 25
N_ITERS_PER_PLOT = 25
POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None

SIG_ETA_TRAIN = np.diag([1e-1, 1e-1])

# input inference
INFERENCE = GaussianI2c(
    inference=Linearize(),
    Q=None,
    R=np.diag([1.0]),
    Qf=None,
    alpha=1e9,
    alpha_update_tol=1.0,  # constant
    mu_u=np.zeros((N_DURATION, 1)),
    sig_u=1e2 * np.eye(1),
    mu_x_term=np.array([[-5, -5]]),
    sig_x_term=np.diag([2e0, 2e0]),
)
