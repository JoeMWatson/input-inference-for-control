"""
Types for defining experiment hyperparameters
Includes quadrature types and initialization
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class GaussianI2c(object):

    inference: dataclass
    alpha: float
    alpha_update_tol: float
    Q: np.ndarray
    Qf: np.ndarray
    R: np.ndarray
    mu_u: np.ndarray
    sig_u: np.ndarray
    mu_x_term: np.ndarray
    sig_x_term: np.ndarray


# Inference
@dataclass
class Linearize(object):
    """No parameters"""


@dataclass
class CubatureQuadrature(object):
    alpha: float
    beta: float
    kappa: float

    @staticmethod
    def pts(dim):
        return np.concatenate((np.zeros((1, dim)), np.eye(dim), -np.eye(dim)), axis=0)

    def weights(self, dim):
        assert self.alpha > 0
        lam = self.alpha ** 2 * (dim + self.kappa) - dim
        sf = np.sqrt(dim + lam)
        w0_sig_extra = 1 - self.alpha ** 2 + self.beta
        weights_mu = (1 / (2 * (dim + lam))) * np.ones((1 + 2 * dim,))
        weights_mu[0] = 2 * lam * weights_mu[0]
        weights_sig = np.copy(weights_mu)
        weights_sig[0] += w0_sig_extra
        return sf, weights_mu, weights_sig


@dataclass
class GaussHermiteQuadrature(object):
    degree: int

    def __post_init__(self):
        assert self.degree >= 1
        self.gh_pts, self.gh_weights = np.polynomial.hermite.hermgauss(self.degree)

    def pts(self, dim):
        grid = np.meshgrid(*(self.gh_pts,) * dim)
        return np.vstack(tuple(map(np.ravel, grid))).T

    def weights(self, dim):
        grid = np.meshgrid(*(self.gh_weights,) * dim)
        w = np.vstack(tuple(map(np.ravel, grid))).T
        w = np.prod(w, axis=1) / (np.pi ** (dim / 2))
        return np.sqrt(2), w, w
