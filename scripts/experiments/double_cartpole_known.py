import numpy as np
from pi2c.exp_types import Pi2c

ENVIRONMENT = "DoubleCartpoleKnown"  # environment to control


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
  Q=np.diag([1.0, 1.0, 100.0, 1.0, 100.0, 1.0, 1.0, 1.0]),
  R=np.diag([0.1]),
  ALPHA=90.,
  alpha_update_tol=0.9995,
  SIG_U=0.04 * np.eye(1),
  msg_iter=1,
  msg_tol=1e-3,
  em_tol=1e-3,
  backwards_contraction=None,
)
