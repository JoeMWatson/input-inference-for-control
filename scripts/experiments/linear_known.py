import numpy as np
from pi2c.exp_types import Pi2c

ENVIRONMENT = "LinearKnown"  # environment to control

# top level training parameters
N_DURATION = 60
N_EPISODE = 1
N_INFERENCE = 1
N_AUG = 0
N_STARTING = 0
N_ITERS_PER_PLOT = N_INFERENCE + 1
POLICY_COVAR =  0 * np.eye(1)

# model learning
MODEL = None

# input inference
INFERENCE = Pi2c(
  Q=np.diag([10.0, 10.0]),
  R=np.diag([1.0]),
  ALPHA=300.,
  alpha_update_tol=0.0,
  SIG_U=100. * np.eye(1),
  msg_iter=100,
  msg_tol=1e-3,
  em_tol=1e-3,
  backwards_contraction=None
)