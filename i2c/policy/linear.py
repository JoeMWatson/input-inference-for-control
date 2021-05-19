"""
Time-varying linear Gaussian Controllers
"""

import numpy as np
from numpy.random import multivariate_normal as mvn


class TimeIndexedLinearGaussianPolicy(object):
    def __init__(self, sig_u, H, dim_u, dim_x, control_step=1):
        self.H, self.dim_u, self.dim_x = H, dim_u, dim_x
        self.sig_u = sig_u
        self.control_step = control_step
        self.init()

    def init(self):
        self.K = np.zeros((self.H, self.dim_u, self.dim_x))
        self.k = np.zeros((self.H, self.dim_u))
        self.sig_k = np.repeat(self.sig_u[None, :, :], self.H, axis=0)

    def zero(self):
        self.K = np.zeros((self.H, self.dim_u, self.dim_x))
        self.k = np.zeros((self.H, self.dim_u))
        self.sig_k = np.zeros((self.H, self.dim_u, self.dim_u))

    def write(self, K, k, sig_k):
        self.K[:, :, :] = K[:, :, :]
        self.k[:, :] = k[:, :]
        self.sig_k[:, :, :] = sig_k[:, :]

    def __call__(self, i, x, deterministic=True):
        assert i < self.H
        if i % self.control_step == 0:
            kx = self.K[i, :, :] @ x
            mu = kx + self.k[i, :, None]
            sig = self.sig_k[i, :, :]

            if deterministic:
                self.u = mu
            else:
                self.u = mvn(mu[:, 0], sig, 1)

        return self.u


class ExpertTimeIndexedLinearGaussianPolicy(object):

    hard_exp_threshold = 3.0

    def __init__(self, sig_u, H, dim_u, dim_x, soft=True):
        self.H, self.dim_u, self.dim_x = H, dim_u, dim_x
        self.sig_u = sig_u
        self.soft = soft
        self.init()

    def init(self):
        self.K = np.zeros((self.H, self.dim_u, self.dim_x))
        self.k = np.zeros((self.H, self.dim_u))
        self.sig_k = np.repeat(self.sig_u[None, :, :], self.H, axis=0)
        self.mu = np.zeros((self.H, self.dim_x))
        self.lam = np.ones((self.H, self.dim_x, self.dim_x))

    def zero(self):
        self.init()

    def write(self, K, k, sig_k, mu, lam):
        self.K[:, :, :] = K[:, :, :]
        self.k[:, :] = k[:, :]
        self.sig_k[:, :, :] = sig_k[:, :, :]
        self.mu[:, :] = mu[:, :]
        self.lam[:, :, :] = lam[:, :, :]

    def __call__(self, i, x, deterministic=True):
        assert i < self.H
        dist = x - self.mu[i, :, None]

        exp = 0.5 * dist.T @ self.lam[i, :, :] @ dist
        if self.soft:
            p = np.exp(-exp).item()
        else:  # hard
            p = float(abs(exp) < self.hard_exp_threshold)

        kx = p * self.K[i, :, :] @ dist
        mu = self.k[i, :, None] + kx

        if deterministic:
            return mu.reshape((self.dim_u, 1))
        else:
            sig = self.sig_k[i, :, :].reshape((self.dim_u, self.dim_u))
            return mu + mvn(np.zeros((self.dim_u,)), sig, 1).reshape((self.dim_u, 1))
