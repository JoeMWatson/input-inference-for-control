import numpy as np
from pi2c.exp_types import Pi2c

ENVIRONMENT = "CartpoleKnown"  # environment to control

# top level training parameters
N_DURATION = 1000
N_EPISODE = 1
N_INFERENCE = 200
N_AUG = 1
N_STARTING = 0
N_PLOTS = 5
N_ITERS_PER_PLOT = N_INFERENCE / N_PLOTS

POLICY_COVAR = 0.0 * np.eye(1)

# model learning
MODEL = None

# input inference
INFERENCE = Pi2c(
  Q=np.diag([1.0, 1.0, 100.0, 1.0, 1.0]),
  R=np.diag([1.0]),
  ALPHA=67.,
  alpha_update_tol=0.993,
  # SIG_U=0.1 * np.eye(1),
  SIG_U=0.25 * np.eye(1),
  msg_iter=1,
  msg_tol=1e-3,
  em_tol=1e-3,
  backwards_contraction=None,
)
