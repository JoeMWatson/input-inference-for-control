"""
Time-varying linear Gaussian Controller
"""

import numpy as np


class TimeIndexedLinearGaussianPolicy(object):

    def __init__(self, sigU, H, dim_u, dim_x):
        self.H, self.dim_u, self.dim_x = H, dim_u, dim_x
        self.K = np.zeros((H, dim_u, dim_x))
        self.k = np.zeros((H, dim_u))
        self.sigk = np.ones((H, dim_u)).dot(sigU)

    def zero(self):
        self.K = np.zeros((self.H, self.dim_u, self.dim_x))
        self.k = np.zeros((self.H, self.dim_u))
        self.sigk = np.zeros((self.H, self.dim_u))

    def __call__(self, i, x):
        assert i < self.H
        kx = self.K[i, :, :].dot(x).squeeze()
        mu = kx + self.k[i, :]
        sig = self.sigk[i, :]
        u = mu + sig * np.random.randn()
        return u
