"""
i2c MPC controllers
"""

import os
import numpy as np
from numpy.random import multivariate_normal as mvn
from copy import deepcopy
import matplotlib.pyplot as plt

# for filtering
from i2c.exp_types import CubatureQuadrature
from i2c.inference.quadrature import QuadratureInference


class MpcPolicy(object):

    def __init__(self, i2c, n_iter, sig_u, z_traj=None):
        self.dim_u, self.dim_x = i2c.sys.dim_u, i2c.sys.dim_x
        self.sig_u = sig_u
        i2c.state_action_independence = True
        i2c.tau = 0
        self.i2c = i2c
        self.i2c_init = deepcopy(i2c)
        self.model = i2c.sys
        self.cell_init = deepcopy(i2c.cells[0])
        self.n_iter = n_iter
        self.z_traj = z_traj
        if z_traj is not None:
            for i, c in enumerate(self.i2c.cells):
                c.z = z_traj[i, :, None]
        self.xu_history = []
        self.z_history = []

    def set_control(self, feedforward):
        if feedforward:
            self.i2c.state_action_independence = True
            self.i2c.tau = 0
        else:
            self.i2c.state_action_independence = False
            self.i2c.tau = self.i2c.H

    def reset(self):
        self.i2c = deepcopy(self.i2c_init)
        self.model = self.i2c.sys
        self.xu_history = []
        self.z_history = []

    def optimize(self, n_iter, x):
        assert x.shape == (self.dim_x, 1), f"{x.shape}, {(self.dim_x, 1)}"
        self.i2c.sys.x0 = x

        for _ in range(n_iter):
            self.i2c._forward_backward_msgs()
            self.i2c._update_priors()

    def __call__(self, i, x, deterministic=True):
        self.optimize(self.n_iter, x)
        self.i2c.compute_update_alpha(True, calc_evar=False, calc_propagate=False)
        self.xu_history.append(self.i2c.get_marginal_state_action())
        self.z_history.append(self.i2c.get_marginal_observed_trajectory()[0])
        mu = np.copy(self.i2c.cells[0].mu_u0_m)
        sig = np.copy(self.i2c.cells[0].sig_u0_m)

        if deterministic:
            u = mu.reshape((self.dim_u, 1))
        else:
            u = mvn(mu.squeeze(), sig, 1).reshape((self.dim_u, 1))

        self.i2c.cells.pop(0)
        self.i2c.cells.append(deepcopy(self.cell_init))
        self.i2c.cells[-1].sys = self.i2c.sys
        if self.z_traj is not None:
            if (i + self.i2c.H) < self.z_traj.shape[0]:
                self.i2c.cells[-1].z = self.z_traj[i + self.i2c.H, :, None]
            else:
                self.i2c.cells[-1].z = self.i2c.cells[-2].z
        return u

    def update_models(self, sys):
        self.i2c.sys = sys
        self.cell_init.sys = sys

    def plot_history(self, res_path, name=""):
        horizon = len(self.xu_history)
        hist = np.asarray(self.xu_history)
        f, a = plt.subplots(self.dim_x + self.dim_u)
        for i, _a in enumerate(a):
            for t in range(horizon):
                _t = np.arange(t, t + self.i2c.H)
                _a.plot(_t, hist[t, :, i])
        plt.savefig(
            os.path.join(res_path, f"xu_history_{name}.png"),
            bbox_inches="tight",
            format="png",
        )
        plt.close(f)

        hist = np.asarray(self.z_history)
        f, a = plt.subplots(self.model.dim_z)
        for i, _a in enumerate(a):
            for t in range(horizon):
                _t = np.arange(t, t + self.i2c.H)
                _a.plot(_t, hist[t, :, i])
        plt.savefig(
            os.path.join(res_path, f"z_history_{name}.png"),
            bbox_inches="tight",
            format="png",
        )
        plt.close(f)


class PartiallyObservedMpcPolicy(MpcPolicy):

    def __init__(self, i2c, n_iter, sig_u, z_traj=None):
        super().__init__(i2c, n_iter, sig_u, z_traj)
        self.mu = i2c.sys.x0
        self.covar = i2c.sys.sig_x0
        self.mus = []
        self.covars = []
        inference = CubatureQuadrature(1, 0, 0)
        self.dyn_inf = QuadratureInference(inference, self.dim_x)
        self.meas_inf = QuadratureInference(inference, self.dim_x)

    def filter(self, y, u):
        assert u.shape == (self.dim_u, 1)
        # pass old belief through dynamics
        # expose quadrature to add control manually
        x_pts = self.dyn_inf.get_x_pts(self.mu, self.covar)
        x_pts = np.concatenate((x_pts, u.T.repeat(x_pts.shape[0], axis=0)), axis=1)
        x_f_pts, _sig_y = self.i2c.sys.forward(x_pts)
        mu_f = np.einsum("b, bi->i", self.dyn_inf.weights_sig, x_f_pts).reshape((-1, 1))
        _sig_f = np.einsum(
            "b,bi,bj->ij", self.dyn_inf.weights_sig, x_f_pts, x_f_pts
        ) - np.outer(mu_f, mu_f)
        sig_eta = np.einsum("b,bij->ij", self.dyn_inf.weights_sig, _sig_y)
        sig_f = _sig_f + sig_eta

        # innovate on new measurement
        mu_y, sig_y = self.meas_inf.forward(self.i2c.sys.measure, mu_f.T, sig_f)
        sig_y += self.i2c.sys.sig_zeta
        K = np.linalg.solve(sig_y.T, self.meas_inf.sig_xy.T).T
        self.mu = mu_f + K @ (y - mu_y)
        self.covar = sig_f - K @ sig_y @ K.T
        return self.mu, self.covar

    def optimize(self, n_iter, mu, covar):
        assert mu.shape == (self.dim_x, 1), f"{mu.shape}, {(self.dim_x, 1)}"
        self.i2c.sys.x0 = mu
        self.i2c.sys.sig_x0 = covar

        for _ in range(n_iter):
            self.i2c._forward_backward_msgs()
            self.i2c._update_priors()

    def __call__(self, i, y, u, deterministic=True):
        if i > 0:
            self.filter(y, u)
        mu = self.mu
        covar = self.covar
        self.mus.append(mu)
        self.covars.append(covar)
        self.optimize(self.n_iter, mu, covar)
        self.xu_history.append(self.i2c.get_marginal_state_action())
        self.z_history.append(self.i2c.get_marginal_observed_trajectory()[0])
        mu = np.copy(self.i2c.cells[0].mu_u0_m)
        sig = np.copy(self.i2c.cells[0].sig_u0_m)

        if deterministic:
            ctrl = mu.reshape((self.dim_u, 1))
        else:
            ctrl = mvn(mu.squeeze(), sig, 1).reshape((self.dim_u, 1))

        self.i2c.cells.pop(0)
        self.i2c.cells.append(deepcopy(self.cell_init))
        self.i2c.cells[-1].sys = self.i2c.sys
        if self.z_traj is not None:
            if (i + self.i2c.H) < self.z_traj.shape[0]:
                self.i2c.cells[-1].z = self.z_traj[i + self.i2c.H, :, None]
            else:
                self.i2c.cells[-1].z = self.i2c.cells[-2].z
        return ctrl
