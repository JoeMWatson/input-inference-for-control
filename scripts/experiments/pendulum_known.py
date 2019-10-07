import numpy as np
from pi2c.exp_types import Pi2c

ENVIRONMENT = "PendulumKnown"  # evironment to control

# top level training parameters
N_DURATION = 100
N_EPISODE = 1
N_INFERENCE = 100
N_AUG = 0
N_STARTING = 0
N_PLOTS = 1
N_ITERS_PER_PLOT = N_INFERENCE / N_PLOTS

POLICY_COVAR = .5 * np.eye(1)

# model learning
MODEL = None

# input inference
INFERENCE = Pi2c(
  Q=np.diag([1, 100.0, 1]),
  R=np.diag([1]),
  ALPHA=100.,
  alpha_update_tol=0.99,
  SIG_U=0.2 * np.eye(1),
  msg_iter=100,
  msg_tol=1e-3,
  em_tol=1e-3,
  backwards_contraction=None,#0.01,
)