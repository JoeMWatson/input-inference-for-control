import numpy as np
from i2c.exp_types import GaussianI2c, CubatureQuadrature

ENVIRONMENT = "DoubleCartpoleKnown"  # environment to control


# top level training parameters
N_DURATION = 1000
N_EPISODE = 1
N_INFERENCE = 200
N_AUG = 1
N_STARTING = 0
N_PLOTS = 10
N_ITERS_PER_PLOT = N_INFERENCE // N_PLOTS

POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None

TAU = 0
# TAU = N_DURATION
SIG_ETA_TRAIN = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
SIG_ETA_TEST = SIG_ETA_TRAIN

# input inference
INFERENCE = GaussianI2c(
    inference=CubatureQuadrature(1, 0, 0),
    Q=np.diag([1.0, 1.0, 100.0, 1.0, 100.0, 10.0, 1.0, 1.0]),
    R=np.diag([0.1]),
    Qf=np.diag([1.0, 1.0, 100.0, 1.0, 100.0, 10.0, 1.0, 1.0]),
    alpha=100.0,
    alpha_update_tol=0.0,
    mu_u=np.zeros((N_DURATION, 1)),
    # alpha_update_tol=1-1e-6,
    # sig_u=0.06 * np.eye(1), # 10 max
    sig_u=0.25 * np.eye(1),
    mu_x_term=None,
    sig_x_term=None,
    # sig_x_term=np.diag([0.01, 0.005, 0.005, 0.05, 0.05, 0.05]), # works but doesnt converge
    # sig_x_term=np.diag([0.5, 0.01, 0.01, 0.2, 0.1, 0.1]), # works but doesnt converge
    msg_iter=1,
    msg_tol=1e-3,
    em_tol=1e-3,
)
