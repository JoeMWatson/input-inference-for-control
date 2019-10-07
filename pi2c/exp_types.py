"""
Types for defining experiment hyperparameters
"""

from collections import namedtuple

# Inference
Pi2c = namedtuple(
    'Pi2c',
    ['ALPHA',
     'alpha_update_tol',
     'Q',
     'R',
     'SIG_U',
     'msg_iter',
     'msg_tol',
     'em_tol',
     'backwards_contraction'])
