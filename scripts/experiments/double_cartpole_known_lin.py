import numpy as np
from i2c.exp_types import GaussianI2c, Linearize

ENVIRONMENT = "DoubleCartpoleKnown"  # environment to control


# top level training parameters
N_DURATION = 250
N_EPISODE = 1
N_INFERENCE = 200
N_AUG = 1
N_STARTING = 0
N_PLOTS = 4
N_ITERS_PER_PLOT = N_INFERENCE // N_PLOTS

N_BUFFER = 0

POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None

sf = 1e-3  # make numbers nice
Q = sf * np.diag([1.0, 1.0, 100.0, 1.0, 100.0, 10.0, 1.0, 1.0])
R = sf * np.diag([0.1])
Qf = sf * np.diag([1.0, 1.0, 100.0, 1.0, 100.0, 10.0, 1.0, 1.0])
# input inference
INFERENCE = GaussianI2c(
    inference=Linearize(),
    Q=Q,
    R=R,
    Qf=Q,
    alpha=0.05,
    alpha_update_tol=0.99,  # 0.99
    mu_u=1e-2 * np.random.randn(N_DURATION, 1),
    sig_u=1.0 * np.eye(1),
    mu_x_term=None,
    sig_x_term=None,
)
