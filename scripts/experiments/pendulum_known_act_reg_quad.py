import numpy as np
from i2c.exp_types import GaussianI2c, CubatureQuadrature

ENVIRONMENT = "PendulumKnownActReg"  # environment to control

# top level training parameters
N_DURATION = 100
N_EPISODE = 1
N_INFERENCE = 300
N_AUG = 0
N_STARTING = 0
N_PLOTS = 4
N_BUFFER = 0
N_ITERS_PER_PLOT = N_INFERENCE / N_PLOTS

POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None

# input inference
INFERENCE = GaussianI2c(
    inference=CubatureQuadrature(1, 0, 0),
    Q=None,
    R=np.diag([1.0]),
    Qf=None,  # np.diag([11, 210., 10.]),
    alpha=300,
    alpha_update_tol=1.0,
    mu_u=np.zeros((N_DURATION, 1)),
    sig_u=0.5 * np.eye(1),  # 5
    mu_x_term=np.array([0.0, 0.0]),
    sig_x_term=np.diag([1e-3, 1e-3]),
)
