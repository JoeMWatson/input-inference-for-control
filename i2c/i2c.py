"""
Implementation of Gaussian input inference for control (i2c)
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal as mvn
import tikzplotlib
import dill

from i2c.exp_types import Linearize, CubatureQuadrature, GaussHermiteQuadrature
from i2c.inference.quadrature import QuadratureInference

# flag to use matplotlib2tikz
PLOT_TIKZ = False

CHECK_COVAR = False  # check the are semi pos def, NOTE really slows things down

DEBUG_PLOTS = False


def moment2information(mu, sigma):
    lam = np.linalg.inv(sigma)
    nu = lam.dot(mu)
    return nu, lam


def information2moment(nu, lam):
    sig = np.linalg.inv(lam)
    mu = sig.dot(nu)
    return mu, sig


def is_pos_def(mat):
    if CHECK_COVAR:
        pd = np.all(np.linalg.eigvals(mat) >= 0.0)
        return pd
    else:
        return True


def concat_normals(mu_1, sig_1, mu_2, sig_2):
    mu = np.concatenate((mu_1, mu_2), axis=0)
    sig = la.block_diag(sig_1, sig_2)
    return mu, sig


class I2cCell(object):
    """A single time index of linear gaussian i2c, like the PGM in the papers"""

    def __init__(
        self,
        i,
        sys,
        mu_u,
        sig_u,
        sig_xi,
        lam_xi,
        sig_xi_terminal,
        mu_x_terminal,
        sig_x_terminal,
        inference,
        dtemp=1.0,
    ):
        """
        :param i: (int) cell index
        :param sys: (object) model of system
        :param mu_u: (np.ndarray) (dim_u,) initial action mean
        :param sig_u: (np.ndarray) (dim_u, dim_u) initial action covariance
        :param sig_xi: (np.ndarray (dim_z, dim_z) cost observation covariance
        :param lam_xi: (np.ndarray (dim_z, dim_z) cost observation precision
        :param sig_xi_terminal: (np.ndarray) (dim_zt, dim_zt) optional terminal cost covariance
        :param mu_x_terminal: (np.ndarray) (dim_x, 1) optimal terminal state mean
        :param sig_x_terminal: (np.ndarray) (dim_x, dim_x) optimal terminal state covariance
        :param inference: (object) inference method
        :param dtemp: (float) per iteration prior annealing rate for nonlinear covariance control
        """
        self.index = i
        self.terminal_cell = False  # modified externally
        self.sys = sys
        self.z = np.copy(self.sys.zg)
        self.z_term = np.copy(self.sys.zg_term)
        self.sig_xi = sig_xi
        self.lam_xi = lam_xi  # often used more than sig_xi

        self.sig_xi_terminal = sig_xi_terminal
        self.mu_x_terminal = mu_x_terminal
        self.sig_x_terminal = sig_x_terminal
        self.mu_u0_base = mu_u
        self.sig_u0_base = np.copy(sig_u)

        self.mu_u0_f, self.sig_u0_f = mu_u, sig_u
        self.mu_u0_m, self.sig_u0_m = mu_u, sig_u
        self.mu_xu0_f, self.sig_xu0_f = concat_normals(
            self.sys.x0, self.sys.sig_x0, self.mu_u0_f, self.sig_u0_f
        )
        self.mu_xu0_m, self.sig_xu0_m = np.copy(self.mu_xu0_f), np.copy(self.sig_xu0_f)

        # inference method
        if isinstance(inference, Linearize):
            self._forward_msgs = self._forward_msgs_linearize
            self._backward_msgs = self._backward_msgs_linearize
            self._calc_likelihood = self._calc_likelihood_linearize
            # TODO is this ok??
            # self._calc_likelihood = self._calc_likelihood_quadrature
            self._propagate_forward = self._propagate_forward_quadrature
            self.prop_dyn_inf = QuadratureInference(
                CubatureQuadrature(1, 0, 0), self.sys.dim_xu
            )
            self.prop_obs_inf = QuadratureInference(
                CubatureQuadrature(1, 0, 0), self.sys.dim_xu
            )
        elif isinstance(inference, (CubatureQuadrature, GaussHermiteQuadrature)):
            self.dyn_inf = QuadratureInference(inference, self.sys.dim_xu)
            self.obs_inf = QuadratureInference(inference, self.sys.dim_xu)
            self.obs_term_inf = QuadratureInference(inference, self.sys.dim_x)
            self.prop_dyn_inf = QuadratureInference(inference, self.sys.dim_xu)
            self.prop_obs_inf = QuadratureInference(inference, self.sys.dim_xu)
            self._forward_msgs = self._forward_msgs_quadrature
            self._backward_msgs = self._backward_msgs_quadrature
            self._calc_likelihood = self._calc_likelihood_quadrature
            self._propagate_forward = self._propagate_forward_quadrature
        else:
            raise ValueError("Unknown inference method")

        self.mu_x3_m_prev = None
        self.sig_x3_m_prev = None

        self.state_action_independence = True
        self.innovate = True

        self.K = np.zeros((self.sys.dim_u, self.sys.dim_x))
        self.k = np.copy(self.mu_u0_base)

        self.sig_eta_pf = np.copy(self.sys.sig_eta)
        self.mu_x3_pf, self.sig_x3_pf = np.copy(self.sys.x0), np.copy(self.sys.sig_x0)
        self.mu_u0_pf = np.copy(self.mu_u0_base)
        self.sig_u0_pf = np.copy(sig_u)

        self.use_expert_controller = True

        # for nonlinear covariance control, we increase the temperature of the
        # prior every iteration by dtemp to converge to the desired terminal
        self.temp = 1.0
        self.dtemp = dtemp

    def _propagate_forward_quadrature(self, mu_x, sig_x):
        self.mu_x0_pf = mu_x
        self.sig_x0_pf = sig_x

        K = np.copy(self.K)
        if self.state_action_independence:
            self.mu_u0_pf = self.mu_u0_m
            self.sig_u0_pf = self.sig_u0_m
        else:
            # marginalize through u (i.e. the deterministic expert controller)
            if self.use_expert_controller:
                try:
                    dist = mvn(self.mu_x0_m.squeeze(), self.sig_x0_m + self.sig_x0_pf)
                    w = dist.pdf(self.mu_x0_pf.squeeze())
                    Z = dist.pdf(self.mu_x0_m.squeeze())
                    K = K * (w / Z)
                except Exception as ex:
                    logging.error(ex)
            self.mu_u0_pf = self.mu_u0_m + K @ (self.mu_x0_pf - self.mu_x0_m)
            self.sig_u0_pf = (
                K @ self.sig_x0_pf @ K.T + self.sig_u0_m - K @ self.sig_x0_m @ K.T
            )

        self.mu_xu0_pf = np.concatenate((self.mu_x0_pf, self.mu_u0_pf), axis=0)
        self.sig_xu0_pf = np.block(
            [
                [self.sig_x0_pf, self.sig_x0_pf @ K.T],
                [K @ self.sig_x0_pf, self.sig_u0_pf],
            ]
        )

        self.mu_z0_pf, self.sig_z0_pf = self.prop_obs_inf.forward(
            self.sys.observe, self.mu_xu0_pf, self.sig_xu0_pf
        )

        # BUT use the closed loop dynamics (inc. controller) for quadrature
        try:
            (
                self.mu_x3_pf,
                sig_x3_p,
                self.sig_eta_pf,
            ) = self.prop_dyn_inf.forward_gaussian(
                self.sys.forward, self.mu_xu0_pf, self.sig_xu0_pf
            )
            self.sig_eta_pf = self.sig_eta_pf
            self.sig_x3_pf = sig_x3_p + self.sig_eta_pf
        except:
            raise ValueError(f"Propagation quadrature exception at cell {self.index}")

        return self.mu_x3_pf, self.sig_x3_pf

    def _propagate_forward_linearize(self, mu_x, sig_x):
        self.mu_x0_pf = mu_x
        self.sig_x0_pf = sig_x

        # marginalize through u (i.e. the deterministic controller)
        self.mu_u0_pf = self.K @ self.mu_x0_pf + self.k
        self.sig_u0_pf = self.K @ self.sig_x0_pf @ self.K.T + self.sigK

        # ignore u and just do closed-loop dynamics
        ABK = self.A + self.B @ self.K
        self.mu_x3_pf = ABK @ self.mu_x0_pf + self.B @ self.k + self.a
        self.sig_x3_pf = (
            ABK @ self.sig_x0_pf @ ABK.T + self.B @ self.sigK @ self.B.T + self.sig_eta
        )

        return self.mu_x3_pf, self.sig_x3_pf

    def _propagate_backward(self, mu_x, sig_x):
        if mu_x is None:
            self.mu_x3_pb = self.mu_x3_pf
            self.sig_x3_pb = self.sig_x3_pf
        else:
            self.mu_x3_pb = mu_x
            self.sig_x3_pb = sig_x

        self.mu_xu0_pb = self.mu_xu0_pf + self.J_dyn_p.dot(
            self.mu_x3_pb - self.mu_x3_pf
        )
        self.sig_xu0_pb = self.sig_xu0_pf + self.J_dyn_p.dot(
            self.sig_x3_pb - self.sig_x3_pf
        ).dot(self.J_dyn_p.T)

        self.mu_x0_pb = self.mu_xu0_pb[: self.sys.dim_x]
        self.sig_x0_pb = self.sig_xu0_pb[: self.sys.dim_x, : self.sys.dim_x]
        self.mu_u0_pb = self.mu_xu0_pb[self.sys.dim_x :]
        self.sig_u0_pb = self.sig_xu0_pb[self.sys.dim_x :, self.sys.dim_x :]

        sig_ux = self.sig_xu0_pb[self.sys.dim_x :, : self.sys.dim_x]
        self.Kp = np.linalg.solve(self.sig_x0_pb.T, sig_ux.T).T
        self.kp = self.mu_u0_pb - self.K.dot(self.mu_x0_pb)

        return self.mu_x0_pb, self.sig_x0_pb

    def _forward_msgs_linearize(self, mu0, sig0):
        self.mu_x0_f = mu0
        self.sig_x0_f = sig0
        assert is_pos_def(self.sig_x0_f)

        if self.state_action_independence:
            self.mu_xu0_f, self.sig_xu0_f = concat_normals(
                self.mu_x0_f, self.sig_x0_f, self.mu_u0_f, self.sig_u0_f
            )
        else:
            # why recompute K? just store it
            # compute new joint from new prior
            sig_xx = self.sig_xu0_f[: self.sys.dim_x, : self.sys.dim_x]
            sig_ux = self.sig_xu0_f[self.sys.dim_x :, : self.sys.dim_x]
            K = np.copy(self.K)
            if self.use_expert_controller:
                dist = mvn(
                    self.mu_xu0_f[: self.sys.dim_x, :].squeeze(), sig_xx + self.sig_x0_f
                )
                w = dist.pdf(self.mu_x0_f.squeeze())
                Z = dist.pdf(self.mu_xu0_f[: self.sys.dim_x, :].squeeze())
                K = K * (w / Z)

            self.mu_u0_f = self.mu_u0_m + K @ (self.mu_x0_f - self.mu_x0_m)
            self.sig_u0_f = self.sig_u0_m - K @ sig_ux.T + K @ self.sig_x0_f @ K.T
            # rebuild joint
            self.mu_xu0_f = np.concatenate((self.mu_x0_f, self.mu_u0_f), axis=0)
            self.sig_xu0_f = np.block(
                [
                    [self.sig_x0_f, self.sig_x0_f @ K.T],
                    [K @ self.sig_x0_f, self.sig_u0_f],
                ]
            )

        # innovate state
        self.nu_x0_f, self.lambda_x0_f = moment2information(self.mu_x0_f, self.sig_x0_f)

        # linearize observation model about prior
        self.mu_z0_f, self.E, self.e, self.F = self.sys.observe_linearize(
            self.mu_xu0_f.T
        )
        # required for backwards computation
        self.sig_z1_f = self.sig_xi + self.F @ self.sig_u0_f @ self.F.T
        self.lambda_z1_f = np.linalg.inv(self.sig_z1_f)

        self.nu_z1_f = self.E.T @ (
            self.lambda_z1_f @ (self.z - self.F @ self.mu_u0_f - self.e)
        )
        self.nu_x1_f = self.nu_x0_f + self.nu_z1_f
        self.lambda_x1_f = self.lambda_x0_f + self.E.T @ self.lambda_z1_f @ self.E

        self.mu_x1_f, self.sig_x1_f = information2moment(self.nu_x1_f, self.lambda_x1_f)

        EF = np.hstack((self.E, self.F))
        self.sig_z0_f = EF @ self.sig_xu0_f @ EF.T + self.sig_xi
        sig_zxu_f = EF @ self.sig_xu0_f
        K = np.linalg.solve(self.sig_z0_f.T, sig_zxu_f).T

        self.mu_xu1_f = self.mu_xu0_f + K @ (
            self.z - self.mu_z0_f.reshape((self.sys.dim_z, -1))
        )

        self.sig_xu1_f = self.sig_xu0_f - K @ sig_zxu_f

        self.mu_x1_f = self.mu_xu1_f[: self.sys.dim_x]
        self.sig_x1_f = self.sig_xu1_f[: self.sys.dim_x, : self.sys.dim_x]
        self.mu_u1_f = self.mu_xu1_f[self.sys.dim_x :]
        self.sig_u1_f = self.sig_xu1_f[self.sys.dim_x :, self.sys.dim_x :]

        # for Riccati calc
        self.sig_z2_f = self.sig_xi + self.E @ self.sig_x0_f @ self.E.T
        self.lambda_z2_f = np.linalg.inv(self.sig_z2_f)
        self.nu_z2_f = (
            self.F.T @ self.lambda_z2_f @ ((self.z - self.E @ self.mu_x0_f) - self.e)
        )

        # propagate state and action mean, get linearizations from priors in
        # first pass, use marginals for subsequent passes
        self.mu_x3_f, self.A, self.B, self.a, self.sig_eta = self.sys.forward_linearize(
            self.mu_xu1_f.T
        )
        self.AB = np.hstack((self.A, self.B))

        # propagate uncertainty
        self.sig_x3_f = self.AB @ self.sig_xu1_f @ self.AB.T + self.sig_eta
        # for Riccati
        self.sig_u2_f = self.B @ self.sig_u1_f @ self.B.T  # needed for Riccati
        self.sig_x2_f = self.A @ self.sig_x1_f @ self.A.T + self.sig_eta
        self.lambda_x2_f = np.linalg.inv(self.sig_x2_f)  # needed in Ricatti calc

        # get linearization between state transitions
        self.J_dyn = la.solve(
            self.sig_x3_f.T,
            self.AB @ self.sig_xu1_f,
            check_finite=False,
            assume_a="pos",
        ).T
        self.Jx_dyn = self.J_dyn[: self.sys.dim_x, :]

        sig_xu1_3 = self.AB @ self.sig_xu1_f[:, : self.sys.dim_x]
        # statistical linearization used in smoothing and likelihood computation
        self.Jx_dyn = np.linalg.solve(self.sig_x3_f.T, sig_xu1_3).T

        self.nu_x3_f, self.lambda_x3_f = moment2information(self.mu_x3_f, self.sig_x3_f)
        return self.mu_x3_f, self.sig_x3_f

    def _forward_msgs_quadrature(self, mu0, sig0):
        self.mu_x0_f = mu0
        self.sig_x0_f = sig0
        assert is_pos_def(self.sig_x0_f)

        if self.state_action_independence:
            # equivalent to feedforward
            mu_xu0_ff, sig_xu0_ff = concat_normals(
                self.mu_x0_f, self.sig_x0_f, self.mu_u0_f, self.sig_u0_f
            )
            self.mu_xu0_f, self.sig_xu0_f = mu_xu0_ff, sig_xu0_ff
        else:
            # why recompute K? just store it
            # compute new joint from new prior
            sig_xx = self.sig_xu0_f[: self.sys.dim_x, : self.sys.dim_x]
            sig_ux = self.sig_xu0_f[self.sys.dim_x :, : self.sys.dim_x]
            K = np.copy(self.K)

            # feedback control via conditional update
            dist = mvn(
                self.mu_xu0_f[: self.sys.dim_x, :].squeeze(), sig_xx + self.sig_x0_f
            )
            w = dist.pdf(self.mu_x0_f.squeeze())
            Z = dist.pdf(self.mu_xu0_f[: self.sys.dim_x, :].squeeze())
            K = K * (w / Z)

            # self.mu_u0_f = self.mu_xu0_f[self.sys.dim_x:, :] + K @ self.mu_x0_f
            # self.mu_u0_f = self.k + K @ self.mu_x0_f
            self.mu_u0_f = self.mu_u0_m + K @ (self.mu_x0_f - self.mu_x0_m)
            self.sig_u0_f = self.sig_u0_m - K @ sig_ux.T + K @ self.sig_x0_f @ K.T
            # rebuild joint
            self.mu_xu0_f = np.concatenate((self.mu_x0_f, self.mu_u0_f), axis=0)
            self.sig_xu0_f = np.block(
                [
                    [self.sig_x0_f, self.sig_x0_f @ K.T],
                    [K @ self.sig_x0_f, self.sig_u0_f],
                ]
            )

        # observation
        if self.innovate:
            mu_z0_f, sig_z0_f = self.obs_inf.forward(
                self.sys.observe, self.mu_xu0_f, self.sig_xu0_f
            )
            sig_z0_f += self.sig_xi

            # K is the (conditional) statistical linearization Sig_xy Sig_y^{-1}
            # which is also known as the Kalman gain in this context
            K = la.solve(
                sig_z0_f.T, self.obs_inf.sig_xy.T, check_finite=False, assume_a="pos"
            ).T

            self.mu_xu1_f = self.mu_xu0_f + K @ (self.z - mu_z0_f)
            self.sig_xu1_f = self.sig_xu0_f - K @ self.obs_inf.sig_xy.T
            self.mu_z0_f, self.sig_z0_f = mu_z0_f, sig_z0_f
        else:
            self.mu_xu1_f = self.mu_xu0_f
            self.sig_xu1_f = self.sig_xu0_f

        self.mu_x1_f = self.mu_xu1_f[: self.sys.dim_x]
        self.sig_x1_f = self.sig_xu1_f[: self.sys.dim_x, : self.sys.dim_x]
        self.mu_u1_f = self.mu_xu1_f[self.sys.dim_x :]
        self.sig_u1_f = self.sig_xu1_f[self.sys.dim_x :, self.sys.dim_x :]

        # dynamics
        self.mu_x3_f, sig_x3_f, self.sig_eta = self.dyn_inf.forward_gaussian(
            self.sys.forward, self.mu_xu1_f, self.sig_xu1_f
        )
        self.sig_eta = self.sig_eta
        sig_x3_f += self.sig_eta

        self.sig_x3_f = (sig_x3_f + sig_x3_f.T) / 2

        self.J_dyn = la.solve(
            self.sig_x3_f.T, self.dyn_inf.sig_xy.T, check_finite=False, assume_a="pos"
        ).T
        self.Jx_dyn = self.J_dyn[
            : self.sys.dim_x, :
        ]  # get linearization between state transitions

        if self.terminal_cell and self.sig_xi_terminal is not None:

            mu_z, sig_z = self.obs_term_inf.forward(
                self.sys.observe_terminal_x, self.mu_x3_f, self.sig_x3_f
            )
            sig_z += self.sig_xi_terminal

            # derive K from just the noise
            K = la.solve(
                sig_z.T, self.obs_term_inf.sig_xy.T, check_finite=False, assume_a="pos"
            ).T

            self.mu_x3_f += K @ (self.z_term - mu_z.reshape((-1, 1)))
            self.sig_x3_f -= K @ self.obs_term_inf.sig_xy.T

        self.nu_x3_f, self.lambda_x3_f = moment2information(self.mu_x3_f, self.sig_x3_f)

        return self.mu_x3_f, self.sig_x3_f

    def _backward_msgs_linearize(self, mu_end, sig_end):
        end_in_chain = mu_end is None and sig_end is None
        if end_in_chain:
            # Covariance Control
            if self.sig_x_terminal is not None:
                self.sig_x3_m = self.sig_x_terminal
                # backcalculate sig_xi_terminal (lagrange multiplier)
                mu_z, E, e = self.sys.observe_terminal_linearize(self.mu_x3_f)
                sig_zgx = E @ self.sig_x3_f @ E.T
                sig_zx = E @ self.sig_x3_f
                mp_inv = np.linalg.inv(sig_zx @ sig_zx.T)
                dsig_x = self.sig_x3_f - self.sig_x_terminal
                sig_z = np.linalg.inv(mp_inv @ (sig_zx @ dsig_x @ sig_zx.T) @ mp_inv.T)
                sig_xi_terminal = sig_z - sig_zgx

                if self.mu_x_terminal is None:
                    # calculate it from sig_x_terminal
                    K = la.solve(sig_z, sig_zx, check_finite=False, assume_a="pos").T
                    self.mu_x3_m = self.mu_x3_f + K @ (
                        self.z_term.reshape((-1, 1)) - mu_z.reshape((-1, 1))
                    )
                else:
                    self.mu_x3_m = np.copy(self.mu_x_terminal)

            # Terminal Cost
            elif self.sig_xi_terminal is not None:
                (
                    mu_z,
                    E,
                    e,
                ) = self.sys.observe_terminal_linearize(self.mu_x3_f)
                sig_z = E @ self.sig_x3_f @ E.T + self.sig_xi_terminal

                K = np.linalg.solve(sig_z.T, E @ self.sig_x3_f).T

                self.mu_x3_m = self.mu_x3_f + K @ (self.z_term - mu_z.reshape((-1, 1)))
                self.sig_x3_m = self.sig_x3_f - K @ E @ self.sig_x3_f

                self.lambda_x3_b = np.linalg.inv(self.sig_x3_m) - self.lambda_x3_f
                self.nu_x3_b = (
                    np.linalg.solve(self.sig_x3_m, self.mu_x3_m) - self.nu_x3_f
                )

                sig_xi_terminal = self.sig_xi_terminal

            # No Terminal Adjustment
            else:
                self.mu_x3_m = self.mu_x3_f
                self.sig_x3_m = self.sig_x3_f
                sig_xi_terminal = 1e6 * np.eye(self.sys.dim_x)

            self.mu_z3_m, E, e = self.sys.observe_terminal_linearize(self.mu_x3_m)
            self.sig_z3_m = E @ self.sig_x3_m @ E.T + sig_xi_terminal
        else:
            self.mu_x3_m = mu_end
            self.sig_x3_m = sig_end

        # shortcut state de-innovation via marginal equality and get auxiliary after
        self.lambda_x2_a = self.lambda_x3_f - self.lambda_x3_f @ (
            self.sig_x3_m @ self.lambda_x3_f
        )
        assert is_pos_def(
            self.lambda_x2_a
        ), f"{self.index} {np.linalg.eigvals(self.lambda_x2_a)}"
        self.nu_x2_a = self.nu_x3_f - self.lambda_x3_f @ self.mu_x3_m

        self.mu_xu1_m = self.mu_xu1_f + self.J_dyn @ (self.mu_x3_m - self.mu_x3_f)
        self.sig_xu1_m = (
            self.sig_xu1_f + self.J_dyn @ (self.sig_x3_m - self.sig_x3_f) @ self.J_dyn.T
        )

        # pass through equality node
        self.mu_xu0_m = self.mu_xu1_m
        self.sig_xu0_m = self.sig_xu1_m

        self.mu_x0_m = self.mu_xu0_m[: self.sys.dim_x]
        self.mu_u0_m = self.mu_xu0_m[self.sys.dim_x :]
        self.sig_x0_m = self.sig_xu0_m[: self.sys.dim_x, : self.sys.dim_x]
        self.sig_u0_m = self.sig_xu0_m[self.sys.dim_x :, self.sys.dim_x :]
        sig_ux = self.sig_xu0_m[self.sys.dim_x :, : self.sys.dim_x]

        self.K = np.linalg.solve(self.sig_x0_m.T, sig_ux.T).T

        self.k = self.mu_u0_m - self.K @ self.mu_x0_m
        self.u_pol = self.K @ self.mu_x0_m + self.k
        self.u_pol_K = self.K @ self.mu_x0_m
        self.sigK = self.sig_u0_m - self.K @ sig_ux.T

        # get marginalised observation
        z, C, _, D = self.sys.observe_linearize(self.mu_xu0_m.T)
        self.mu_z0_m = z
        self.sig_z0_m = C @ self.sig_x0_m @ C.T + D @ self.sig_u0_m @ D.T

        return self.mu_x0_m, self.sig_x0_m

    def _backward_msgs_quadrature(self, mu_end, sig_end):
        end_in_chain = mu_end is None and sig_end is None
        if end_in_chain:
            # Covariance Control
            if self.sig_x_terminal is not None:
                self.mu_x3_m = self.mu_x3_f
                sig_x3_f = self.temp * self.sig_x3_f
                sig_x_terminal = self.sig_x_terminal
                self.temp += self.dtemp
                self.sig_x3_m = sig_x3_f - sig_x3_f @ np.linalg.solve(
                    (sig_x_terminal + sig_x3_f), sig_x3_f
                )
                self.mu_x3_m = self.sig_x3_m @ (
                    np.linalg.solve(sig_x3_f, self.mu_x3_f)
                    + np.linalg.solve(sig_x_terminal, self.mu_x_terminal)
                )

            # No Terminal Adjustment
            else:
                self.mu_x3_m = self.mu_x3_f
                self.sig_x3_m = self.sig_x3_f

            # compute terminal observation if it exists
            if self.sig_xi_terminal is not None:
                self.mu_z3_m, self.sig_z3_m = self.obs_term_inf.forward(
                    self.sys.observe_terminal_x, self.mu_x3_m, self.sig_x3_m
                )
            else:
                self.mu_z3_m, self.sig_z3_m = None, None

        else:
            self.mu_x3_m = mu_end
            self.sig_x3_m = sig_end

        self.sig_x_lag_m = self.Jx_dyn @ self.sig_x3_m

        self.mu_xu1_m = self.mu_xu1_f + self.J_dyn @ (self.mu_x3_m - self.mu_x3_f)
        self.sig_xu1_m = (
            self.sig_xu1_f + self.J_dyn @ (self.sig_x3_m - self.sig_x3_f) @ self.J_dyn.T
        )

        # pass through equality node
        self.mu_xu0_m = self.mu_xu1_m
        self.sig_xu0_m = self.sig_xu1_m
        # get terms
        self.mu_x0_m = self.mu_xu0_m[: self.sys.dim_x]
        self.sig_x0_m = self.sig_xu0_m[: self.sys.dim_x, : self.sys.dim_x]
        self.mu_u0_m = self.mu_xu0_m[self.sys.dim_x :]
        self.sig_u0_m = self.sig_xu0_m[self.sys.dim_x :, self.sys.dim_x :]

        self.mu_z0_m, self.sig_z0_m = self.obs_inf.forward(
            self.sys.observe, self.mu_xu1_m, self.sig_xu1_m
        )

        # taking the joint of x and u, p(u|x) is a linear model
        # integrating out the state distribution gives us the controller
        sig_ux = self.sig_xu1_m[self.sys.dim_x :, : self.sys.dim_x]

        self.K = la.solve(
            self.sig_x0_m.T, sig_ux.T, check_finite=False, assume_a="pos"
        ).T
        self.k = self.mu_u0_m - self.K @ self.mu_x0_m
        self.u_pol = self.K @ self.mu_x0_m + self.k
        self.u_pol_K = self.K @ self.mu_x0_m
        self.sigK = self.sig_u0_m - self.K @ sig_ux.T

        return self.mu_x0_m, self.sig_x0_m

    def _backward_ricatti_msgs(self, nu_b, lambda_b):
        """This function was written to verify the CoRL equations for LQR equivalence. It is of limited use."""
        end_in_chain = nu_b is None and lambda_b is None
        if end_in_chain:
            self.nu_x3_b = np.linalg.solve(self.sig_x3_m, self.mu_x3_m) - self.nu_x3_f
            self.lambda_x3_b = np.linalg.inv(self.sig_x3_m) - self.lambda_x3_f
        else:
            self.nu_x3_b = nu_b
            self.lambda_x3_b = lambda_b

        # backwards ricatti equation
        # note: maybe in the future we get the backwards from the forwards and
        # marginal but for now i wanna make sure the maths is correct
        Q = self.E.T @ self.lambda_z1_f @ self.E
        R = self.F.T @ self.lambda_z2_f @ self.F
        Rug = self.nu_z2_f
        nu_u_0 = np.linalg.solve(self.sig_u0_f, self.mu_u0_f)
        gamma = self.lambda_x2_f @ np.linalg.inv(self.lambda_x2_f + self.lambda_x3_b)

        # precision
        ALA = self.A.T @ self.lambda_x3_b @ self.A
        M = np.linalg.inv(self.sig_eta + self.sig_u2_f) + self.lambda_x3_b
        ALMLA = self.A.T @ (
            self.lambda_x3_b @ np.linalg.solve(M, self.lambda_x3_b @ self.A)
        )

        self.lambda_x0_b = Q + ALA - ALMLA

        # mean
        AILM = self.A.T @ (
            np.eye(self.sys.dim_x) - np.linalg.solve(M.T, self.lambda_x3_b.T).T
        )

        self.nu_x0_b = self.nu_z1_f + AILM @ (
            self.nu_x3_b
            - self.lambda_x3_b @ self.a
            - self.lambda_x3_b @ self.B @ self.mu_u1_f
        )

        gamma_L = gamma @ self.lambda_x3_b
        igamma = np.eye(self.sys.dim_x) - gamma
        sig_x3_b = np.linalg.inv(self.lambda_x3_b)
        self.lambda_x2_b = np.linalg.inv(sig_x3_b + self.sig_u2_f)
        self.mu_u2_f = self.B @ self.mu_u1_f
        self.nu_x2_b = self.lambda_x2_b @ sig_x3_b @ self.nu_x3_b - self.mu_u2_f

        psi = gamma_L @ (
            self.sig_x2_f @ (self.lambda_x2_f + np.linalg.inv(sig_x3_b + self.sig_u2_f))
        )

        sig_u = self.sig_u0_m

        K = -sig_u @ self.B.T @ psi @ self.A

        k = sig_u @ (
            nu_u_0
            + Rug
            + self.B.T @ (gamma @ self.nu_x3_b + igamma @ self.nu_x2_b - psi @ self.a)
        )

        self.u_pol = K @ self.mu_x0_m + k
        self.u_pol_K = K @ self.mu_x0_m
        self.K = K
        self.k = k
        self.sigK = self.sig_u0_m

        return self.nu_x0_b, self.lambda_x0_b

    def expected_observation_covar(self):
        dist = self.z - self.mu_z0_m.reshape((-1, 1))
        e_sig_z = np.outer(dist, dist) + self.sig_z0_m
        return e_sig_z

    def expected_propagated_observation_covar(self):
        dist = self.z - self.mu_z0_pf.reshape((-1, 1))
        e_sig_z = np.outer(dist, dist) + self.sig_z0_pf
        return e_sig_z

    def _calc_likelihood_quadrature(self):
        mu_xu, sig_xu = concat_normals(
            self.mu_x0_m, self.sig_x0_m, self.mu_u0_m, self.sig_u0_m
        )
        mu_x, sig_x, sig_eta = self.dyn_inf.forward_gaussian(
            self.sys.forward, mu_xu, sig_xu
        )
        _sig_lag = self.Jx_dyn @ self.sig_x3_m
        _M11 = np.outer(self.mu_x3_m, self.mu_x3_m) + self.sig_x3_m
        _M01 = np.outer(mu_x, self.mu_x3_m) + self.sig_x_lag_m
        _M00 = np.outer(mu_x, mu_x) + sig_x
        _LL_xu = np.linalg.solve(sig_eta, _M00 - _M01 - _M01.T + _M11)

        sig_z = self.expected_observation_covar()
        _LL_z = self.lam_xi @ sig_z
        return _LL_xu, _LL_z

    def _calc_likelihood_linearize(self):
        mu_x = self.A @ self.mu_x0_m + self.B @ self.mu_u0_m + self.a
        sig_x = self.A @ self.sig_x0_m @ self.A.T + self.B @ (self.sig_u0_m @ self.B.T)
        _sig_lag = self.Jx_dyn @ self.sig_x3_m
        _M11 = np.outer(self.mu_x3_m, self.mu_x3_m) + self.sig_x3_m
        _M01 = np.outer(mu_x, self.mu_x3_m) + _sig_lag
        _M00 = np.outer(mu_x, mu_x) + sig_x
        _LL_xu = np.linalg.solve(self.sig_eta, _M00 - _M01 - _M01.T + _M11)

        sig_z = self.expected_observation_covar()
        _LL_z = self.lam_xi @ sig_z
        return _LL_xu, _LL_z

    @staticmethod
    def are_nan(*args):
        are_nan = False
        for a in args:
            are_nan = are_nan or np.any(np.isnan(a))
        return are_nan

    def get_obs_covar(self):
        err = self.mu_z0_m - self.z
        return err @ err.T + self.sig_z0_m


class I2cGraph(object):
    """Manages Gaussian i2c for a whole trajectory"""

    def __init__(
        self,
        sys,
        horizon,
        Q,
        R,
        Qf,
        alpha,
        alpha_update_tol,
        mu_u,
        sig_u,
        mu_x_terminal,
        sig_x_terminal,
        inference,
        res_dir=None,
    ):
        """Graph of Gaussian i2c, made up for 'cells' for each timestep.
        param: sys: dynamics model object
        param: horizon: (int) planning horizon
        param: Q: (np.ndarray) quadratic state cost weight. QR is (dim_z, dim_z)
        oaram: R: (np.ndarray) quadratic action cost weight. QR is (dim_z, dim_z)
        param: Qf: (np.ndarray) quadratic terminal state cost weight
        param: alpha: (float) cost temperature initialization
        param: alpha_update_tol: (float) [0, 1] fraction limiter during M step for regularizer. 0 = no reg
        param: mu_u: (np.ndarray) vector (horizon, dim_u) to initialize u. Used to randomly initialize actions
        param: sig_u: (np.ndarray) matrix (dim_u, dim_u) for action covariance prior
        param: mu_x_terminal: (np.ndarray) optional terminal mean state for covariance control. None if not wanted
        param: sig_x_terminal: (np.ndarray) optional terminal covariance for covariance control. None if not wanted
        param: inference: (object) inference procedure, e.g. linearization, quadrature
        param: res_dir: (string) directory for saving artefacts
        """
        # NOTE: the alpha here is 1/alpha in the paper! it is easier to reason about
        self.sys = sys  # dynamics model
        self.H = horizon
        self.z = np.copy(self.sys.zg)
        self.z_term = np.copy(self.sys.zg_term)
        self.alpha_base = alpha
        self.alpha = self.alpha_base
        self.alphas = [self.alpha]
        self.alphas_desired = [self.alpha]
        self.alphas_pf = [self.alpha]
        self.alpha_risk = []
        self.alpha_update_tol = alpha_update_tol
        self.Q = Q
        self.R = R
        self.inference = inference
        if Q is not None:
            self.QR = la.block_diag(Q, R)
        else:
            self.QR = R
        self.lam_xi0 = np.copy(self.QR)
        self.sig_xi0 = np.linalg.inv(self.QR)
        if Qf is not None:
            self.Qf = Qf
            self.sig_xi_terminal_base = np.linalg.inv(self.Qf)  # R isn't used
        else:
            logging.info("Qf is None")
            self.Qf = np.zeros((self.sys.dim_x, self.sys.dim_x))
            self.sig_xi_terminal_base = None

        self.mu_x0_pf = np.copy(self.sys.x0)
        self.sig_x0_pf = np.copy(self.sys.sig_x0)
        self.mu_x_terminal = (
            mu_x_terminal.reshape((self.sys.dim_x, 1))
            if mu_x_terminal is not None
            else None
        )
        self.sig_x_terminal = sig_x_terminal
        det_sig_xi0 = np.linalg.det(self.sig_xi0)
        assert det_sig_xi0 > 0.0, f"|{self.sig_xi0}| = {det_sig_xi0}"
        lam_xi = np.linalg.inv(self.sig_xi)

        self.cells = [
            I2cCell(
                i,
                sys,
                mu_u[i, :, None],
                sig_u,
                self.sig_xi,
                lam_xi,
                self.sig_xi_terminal,
                self.mu_x_terminal,
                self.sig_x_terminal,
                inference,
            )
            for i in range(self.H)
        ]
        self.cells[-1].terminal_cell = True
        self.reset_metrics(False)
        self.costs_m_all = []
        self.costs_p_all = []
        self.costs_pf_all = []
        self.cost_diff_norms = []
        self.likelihoods_all = []
        self.likelihoods_xu_all = []

        self._propagate = False
        # TODO keep?
        self.tau = horizon - 1

        self.alpha_sigma = 0
        # for saving results
        self.res_dir = res_dir

        if isinstance(inference, (CubatureQuadrature, GaussHermiteQuadrature)):
            self.obs_inf = QuadratureInference(inference, self.sys.dim_xu)
        elif isinstance(inference, Linearize):
            self.obs_inf = QuadratureInference(
                CubatureQuadrature(1, 0, 0), self.sys.dim_xu
            )
        else:
            raise ValueError

        self.policy_valid = False

    def close(self):
        pass

    @property
    def sig_xi(self):
        return self.alpha * self.sig_xi0

    @property
    def sig_xi_terminal(self):
        if self.sig_xi_terminal_base is None:
            return None
        else:
            return self.alpha * self.sig_xi_terminal_base

    def _forward_kalman(self):
        mu = np.copy(self.sys.x0)
        sig = np.copy(self.sys.sig_x0)
        for c in self.cells:
            mu, sig = c._forward_kalman(mu, sig)

    def _backward_kalman(self):
        mu = None
        sig = None
        for c in reversed(self.cells):
            mu, sig = c._backward_kalman(mu, sig)

    def _forward_msgs(self):
        mu = np.copy(self.sys.x0)
        sig = np.copy(self.sys.sig_x0)
        for c in self.cells:
            mu, sig = c._forward_msgs(mu, sig)

    def _backward_msgs(self):
        mu = None
        sig = None
        for c in reversed(self.cells):
            mu, sig = c._backward_msgs(mu, sig)

    def _backward_ricatti_msgs(self):
        nu_b = None
        lambda_b = None
        for c in reversed(self.cells):
            nu_b, lambda_b = c._backward_ricatti_msgs(nu_b, lambda_b)
        self.policy_valid = True

    def calibrate_alpha(self, only_decrease=False):
        # we often want to fix alpha to a value defined by the initial state distribution
        # this function runs propagate and updates alpha based on that,
        # overriding any update limit
        assert self._propagate
        self.propagate()
        z_covar_pf = sum(
            [c.expected_propagated_observation_covar() for c in self.cells]
        )
        alpha_pf_update = self.calculate_alpha(z_covar_pf)
        logging.info(
            f"calibrating alpha from propagation {self.alpha}->{alpha_pf_update}"
        )
        # in the finite horizon settings we only want alpha to go down otherwise we may be very suboptimal
        update = alpha_pf_update < self.alpha if only_decrease else True
        if update:
            self._override_alpha(alpha_pf_update)

    def calculate_alpha(self, z_covar, z_covar_term=None):
        tr = np.trace(self.QR @ z_covar)
        sf = float(self.sys.dim_z * self.H)
        if z_covar_term is not None:
            tr += np.trace(self.Qf @ z_covar_term)
            sf += float(self.sys.dim_z_term)
        return tr / sf

    def compute_update_alpha(self, update_alpha):

        z_covar = self.get_z_covar()
        if self.sig_xi_terminal_base is not None:  # TODO rationalize
            z_covar_term = self.get_z_terminal_covar()
        else:
            z_covar_term = None

        if self.sig_xi_terminal_base is not None:
            alpha_update = self.calculate_alpha(z_covar, z_covar_term)
        else:
            alpha_update = self.calculate_alpha(z_covar)

        if self._propagate:
            z_covar_pf = sum(
                [c.expected_propagated_observation_covar() for c in self.cells]
            )
            alpha_pf_update = self.calculate_alpha(z_covar_pf)
            self.alphas_pf.append(alpha_pf_update)

        self.alphas_desired.append(alpha_update)

        if update_alpha:
            self.update_alpha(alpha_update)

        self.alphas.append(self.alpha)

    def update_alpha(self, alpha_update):

        if np.isnan(alpha_update):
            raise ValueError("Alpha is NaN")
        else:
            alpha_update_ratio = alpha_update / self.alpha
            if self.alpha_update_tol >= 0.0:
                self.alpha_update_tol_u = 2.0 - self.alpha_update_tol
                if alpha_update_ratio < self.alpha_update_tol:
                    alpha_update = self.alpha_update_tol * self.alpha
                if alpha_update_ratio > self.alpha_update_tol_u:
                    alpha_update = self.alpha_update_tol_u * self.alpha
            else:
                alpha_update = self.alpha

            self._update_alpha(alpha_update)

    def _update_alpha(self, update):
        self.alpha = update
        self.lam_xi = np.linalg.inv(self.sig_xi)
        self.update_xi(self.sig_xi, self.lam_xi, self.sig_xi_terminal)

    def _override_alpha(self, update):
        self.alphas[-1] = update
        self.alpha = update
        self.lam_xi = np.linalg.inv(self.sig_xi)
        self.update_xi(self.sig_xi, self.lam_xi, self.sig_xi_terminal)

    def update_xi(self, sig_xi, lam_xi, sig_xi_terminal):
        for c in self.cells:
            c.sig_xi = np.copy(sig_xi)
            c.lam_xi = np.copy(lam_xi)
            if sig_xi_terminal is not None:
                c.sig_xi_terminal = np.copy(sig_xi_terminal)

    def get_z_covar(self):
        return sum([c.expected_observation_covar() for c in self.cells])

    def get_z_propagated_covar(self):
        return sum([c.expected_propagated_observation_covar() for c in self.cells])

    def get_z_terminal_covar(self):
        c = self.cells[-1]
        dist = self.z_term - c.mu_z3_m.reshape((-1, 1))
        return np.outer(dist, dist) + c.sig_z3_m

    def update_models(self):
        pass

    @property
    def propagate_cost_improved(self):
        if len(self.costs_pf) > 1:
            return self.costs_pf[-1] <= self.costs_pf[-2]
        else:  # no reference so yes
            return True

    def _maximize(self):

        self.calc_cost()

        self._update_priors()

        self.compute_update_alpha(update_alpha=True)

        if self.sig_x_terminal is not None and self.mu_x_terminal is not None:
            kl_term = self.mvn_kl_divergence(
                self.cells[-1].mu_x3_pf,
                self.cells[-1].sig_x3_pf,
                self.mu_x_terminal,
                self.sig_x_terminal,
            )
            self.kl_terms.append(kl_term)

        self.policy_entropy.append(self.calc_policy_entropy())
        self.sig_eta_entropy.append(self.calc_sig_eta_entropy())
        self.sig_eta_pf_entropy.append(self.calc_sig_eta_pf_entropy())
        self.x_prior_entropy.append(self.calc_sig_x_prior_entropy())
        self.x_prior_neg_entropy.append(-self.calc_sig_x_prior_entropy())
        if self._propagate:
            self.propagate_entropy.append(self.calc_propagate_entropy())

    def compute_cost(self, x, u):
        y = self.sys.observe(x, u).T
        err = y - self.z
        return np.asscalar(err.T @ self.QR @ err)

    def compute_cost_gaussian(self, mu_xu, sig_xu):
        mu_z, sig_z = self.obs_inf.forward(self.sys.observe, mu_xu, sig_xu)
        err = mu_z - self.z
        sig_z_qr = sig_z @ self.QR
        qr_sig_z_qr = self.QR @ sig_z_qr
        tr_sig_z_qr = np.trace(sig_z_qr)
        x_Q_x = err.T @ self.QR @ err
        m = x_Q_x + tr_sig_z_qr
        v = 2 * np.trace(sig_z_qr @ sig_z_qr) + 4 * err.T @ qr_sig_z_qr @ err
        return np.asscalar(m), np.asscalar(v)

    def calc_cost(self):

        cost_m = [
            self.compute_cost_gaussian(c.mu_xu0_m, c.sig_xu0_m) for c in self.cells
        ]
        mu_cost_m, var_cost_m = zip(*cost_m)

        self.costs_m.append(sum(mu_cost_m))
        self.costs_m_var.append(sum(var_cost_m))

        if self._propagate:
            cost_pf = [
                self.compute_cost_gaussian(c.mu_xu0_pf, c.sig_xu0_pf)
                for c in self.cells
            ]
            mu_cost_pf, var_cost_pf = zip(*cost_pf)
            self.costs_pf.append(sum(mu_cost_pf))
            self.costs_pf_var.append(sum(var_cost_pf))
            self.cost_pf_min.append(min(mu_cost_pf))
        else:
            self.costs_pf.append(-1.0)

    @property
    def cost_pf_entropy(self):
        const = 2 * np.pi * np.e
        return 0.5 * np.log(const * np.asarray(self.costs_pf_var))

    def calc_policy_entropy(self):
        const = 2 * np.pi * np.e
        for i, c in enumerate(self.cells):
            det = np.linalg.det(c.sig_u0_m)
            if det <= 0.0:
                raise ValueError(f"{i}, |{c.sig_u0_m}| = {det}")

        return sum(
            [0.5 * np.log(np.linalg.det(const * c.sig_u0_m)) for c in self.cells]
        )

    def calc_sig_eta_entropy(self):
        const = 2 * np.pi * np.e
        self.sig_eta_entropies = [
            0.5 * np.log(np.linalg.det(const * c.sig_eta)) for c in self.cells
        ]

        return sum(self.sig_eta_entropies)

    def calc_sig_eta_pf_entropy(self):
        const = 2 * np.pi * np.e
        self.sig_eta_pf_entropies = [
            0.5 * np.log(np.linalg.det(const * c.sig_eta_pf)) for c in self.cells
        ]
        return sum(self.sig_eta_pf_entropies)

    def calc_sig_x_pf_entropy_max(self):
        sig_x_pf_det = np.asarray([np.linalg.det(c.sig_x3_pf) for c in self.cells])
        t = np.argmax(sig_x_pf_det)
        return np.max(sig_x_pf_det), self.cells[t].sig_x3_eta, t

    def calc_sig_eta_pf_entropy_max(self):
        sig_eta_pf_det = np.asarray([np.linalg.det(c.sig_eta_pf) for c in self.cells])
        t = np.argmax(sig_eta_pf_det)
        return np.max(sig_eta_pf_det), self.cells[t].sig_eta_pf, t

    def calc_sig_eta_entropy_max(self):
        sig_eta_det = np.asarray([np.linalg.det(c.sig_eta) for c in self.cells])
        t = np.argmax(sig_eta_det)
        return np.max(sig_eta_det), self.cells[t].sig_eta, t

    def calc_sig_eta_bound_check(self):
        sf = 1.1
        sig_eta_pf_det = np.asarray([np.linalg.det(c.sig_eta_pf) for c in self.cells])
        sig_eta_det = np.asarray([np.linalg.det(c.sig_eta) for c in self.cells])
        t = np.argmax(sig_eta_pf_det)
        return (
            np.all(sf * sig_eta_det[: t + 1] < np.max(sig_eta_pf_det)),
            self.cells[t].sig_eta_pf,
        )

    def calc_sig_x_prior_entropy(self):
        const = 2 * np.pi * np.e
        return sum(
            [0.5 * np.log(np.linalg.det(const * c.sig_x3_f)) for c in self.cells]
        )

    def calc_propagate_entropy(self):
        const = 2 * np.pi * np.e
        return sum(
            [0.5 * np.log(np.linalg.det(const * c.sig_x3_pf)) for c in self.cells]
        )

    def _calc_likelihood(self):
        # constant
        ll_const = -0.5 * self.H * (self.sys.dim_x + self.sys.dim_z) * np.log(2 * np.pi)
        # normalising terms
        ll_sig_xi = -0.5 * self.H * np.linalg.det(self.sig_xi)
        ll_sig_eta = -0.5 * sum([np.linalg.det(c.sig_eta) for c in self.cells])
        ll_sig_x0 = -0.5 * np.linalg.det(self.sys.sig_x0)
        # trajectory
        _m_xuz = [c._calc_likelihood() for c in self.cells]
        ll_xu = -0.5 * np.trace(sum([xuz[0] for xuz in _m_xuz]))
        ll_z = -0.5 * np.trace(sum([xuz[1] for xuz in _m_xuz]))
        # initial state
        dist_x0 = self.cells[0].mu_x0_m - self.cells[0].mu_x0_f
        ll_mu_x0 = -0.5 * np.trace(
            np.linalg.solve(
                self.sys.sig_x0, np.outer(dist_x0, dist_x0) + self.cells[0].sig_x0_m
            )
        )

        ll_state_action = ll_sig_eta + ll_xu
        ll_cost = ll_sig_xi + ll_z
        ll_total = ll_const + ll_cost + ll_state_action + ll_sig_x0 + ll_mu_x0
        return ll_total, ll_state_action, ll_cost, ll_xu

    def calc_likelihood(self):
        ll, ll_xu, ll_z, ll_xu = self._calc_likelihood()
        self.likelihoods.append(ll)
        self.likelihoods_xu.append(ll_xu)
        self.likelihoods_z.append(ll_z)
        self.risk.append(-2 * ll_xu / self.alpha)

    def likelihood_z_minima(self, n_min, n_steps):
        return self.list_minima(self.likelihoods_z, n_min, n_steps)

    def likelihood_xu_minima(self, n_min, n_steps):
        return self.list_minima(self.likelihoods_xu, n_min, n_steps)

    @staticmethod
    def list_minima(list_, n_min, n_steps):
        if len(list_) > n_min:  # TODO allow warm up with new model
            if len(list_) > n_steps and n_steps > 0:
                going_down = True
                for i in range(n_steps):
                    going_down = going_down and (list_[-1 - i] < list_[-2 - i])
                minima = going_down
                return minima
            else:
                return False

    @staticmethod
    def indexed_confidence_bound(mu, sig, idx):
        std = 2.0 * np.sqrt(sig[:, idx, idx])
        lower = mu[:, idx] - std
        upper = mu[:, idx] + std
        return upper, lower

    def get_state_action_prior(self):
        return np.asarray([c.mu_xu0_f for c in self.cells])

    def get_propagated_state_action(self):
        return (
            np.asarray([c.mu_xu0_pf for c in self.cells])[:, :, 0],
            np.asarray([c.sig_xu0_pf for c in self.cells]),
        )

    def get_state_and_action(self):
        x = np.asarray([c.mu_x0_m for c in self.cells])
        u = np.asarray([c.mu_u0_m for c in self.cells])
        return x, u

    def get_propagated_state(self):
        mu_x = np.asarray([c.mu_x0_pf for c in self.cells])
        sig_x = np.asarray([c.sig_x0_pf for c in self.cells])
        return mu_x, sig_x

    def _update_priors(self):
        for c in self.cells:
            if c.index <= self.tau and self.tau > 0:
                c.state_action_independence = False

            c.mu_u0_f = np.copy(c.mu_u0_m)
            c.sig_u0_f = np.copy(c.sig_u0_m)

            c.mu_xu0_f_prev = np.copy(c.mu_xu0_f)
            c.sig_xu0_f_prev = np.copy(c.sig_xu0_f)
            c.mu_xu0_f = np.copy(c.mu_xu0_m)
            c.sig_xu0_f = np.copy(c.sig_xu0_m)

    @staticmethod
    def mvn_kl_divergence(mu1, sig1, mu2, sig2):
        mu_diff = mu2 - mu1
        dist = np.asscalar(mu_diff.T @ np.linalg.solve(sig2, mu_diff))
        log_det_ratio = np.log(np.linalg.det(sig2) / np.linalg.det(sig1))
        trace_ratio = np.trace(np.linalg.solve(sig2, sig1))
        return 0.5 * (log_det_ratio + trace_ratio + dist - mu1.shape[0])

    def _forward_backward_msgs(self):
        """Note, there has been experiments with multiple msg iterations
        (like AICO), but so far the added computation hasn't been justified
        in results. Better to just have better priors."""
        self._forward_msgs()
        self._backward_msgs()

    def learn_msgs(self):
        self.em_iter += 1
        # E Step
        self._forward_backward_msgs()
        if self._propagate:
            self.propagate()
        # M Step
        self._maximize()

    def propagate(self):
        mu = np.copy(self.sys.x0)
        sig = np.copy(self.sys.sig_x0)
        for c in self.cells:
            mu, sig = c._propagate_forward(mu, sig)

    def get_local_linear_policy(self):
        """."""
        K = np.asarray([np.copy(c.K) for c in self.cells]).reshape(
            (self.H, self.sys.dim_u, self.sys.dim_x)
        )
        k = np.asarray([np.copy(c.k) for c in self.cells]).reshape(
            (self.H, self.sys.dim_u)
        )
        sig_k = np.asarray([np.copy(c.sigK) for c in self.cells]).reshape(
            (self.H, self.sys.dim_u, self.sys.dim_u)
        )
        return K, k, sig_k

    def get_local_expert_linear_policy(self):
        """Returns params for local linear expert policy."""
        K = np.asarray([np.copy(c.K) for c in self.cells]).reshape(
            (self.H, self.sys.dim_u, self.sys.dim_x)
        )
        k = np.asarray([np.copy(c.mu_u0_m) for c in self.cells]).reshape(
            (self.H, self.sys.dim_u)
        )
        sig_k = np.asarray([np.copy(c.sigK) for c in self.cells]).reshape(
            (self.H, self.sys.dim_u, self.sys.dim_u)
        )
        mu = np.asarray([np.copy(c.mu_x0_m) for c in self.cells]).reshape(
            (
                self.H,
                self.sys.dim_x,
            )
        )
        lam = np.asarray([np.linalg.inv(c.sig_x0_m) for c in self.cells]).reshape(
            (self.H, self.sys.dim_x, self.sys.dim_x)
        )
        return K, k, sig_k, mu, lam

    def get_marginal_input(self):
        return np.asarray([c.mu_u0_m for c in self.cells])

    def get_marginal_state_action(self):
        return np.asarray([c.mu_xu0_m for c in self.cells])

    def get_prior_state_action_distribution(self):
        return (
            np.asarray([c.mu_xu0_f_prev for c in self.cells])[:, :, 0],
            np.asarray([c.sig_xu0_f_prev for c in self.cells]),
        )

    def get_marginal_state_action_distribution(self):
        return (
            np.asarray([c.mu_xu0_m for c in self.cells])[:, :, 0],
            np.asarray([c.sig_xu0_m for c in self.cells]),
        )

    def get_marginal_trajectory(self):
        x = np.asarray([c.mu_x0_m for c in self.cells])
        u = np.asarray([c.mu_u0_m for c in self.cells])
        s = np.hstack((x, u)).reshape((-1, self.sys.dim_xu))
        return s

    def get_marginal_observed_trajectory(self):
        z = [c.mu_z0_m for c in self.cells]
        return np.asarray(z).reshape((-1, self.sys.dim_z)), self.cells[-1].mu_z3_m

    def reset_priors(self):
        self.alpha = self.alpha_base
        for c in self.cells:

            c.mu_u0_f = np.copy(c.mu_u0_base)
            c.sig_u0_f = np.copy(c.sig_u0_base)
            c.sig_xi = self.sig_xi
            # signal for linearizations to be recalculated in during prior
            c.linearized = False
            c.temp = 1.0

        self.reset_metrics()

    def reset_metrics(self, extend=True):
        if extend:
            self.costs_m_all.extend(self.costs_m)
            self.costs_p_all.extend(self.costs_p)
            self.costs_pf_all.extend(self.costs_pf)
            self.likelihoods_all.extend(self.likelihoods)
            self.likelihoods_xu_all.extend(self.likelihoods_xu)

        self.likelihoods = []
        self.likelihoods_xu = []
        self.likelihoods_z = []
        self.em_likelihoods = []
        self.outer_likelihoods = []
        self.state_action_divergence = []
        self.msg_state_divergence = []
        self.msg_state_divergence_converge = []
        self.em_cost = []
        self.costs_p = []
        self.costs_m = []
        self.costs_m_var = []
        self.costs_pf = []
        self.costs_pf_var = []
        self.costs_pf_upper = []
        self.cost_pf_min = []
        self.costs_s = []
        self.traj_gaps = []
        self.policy_entropy = []
        self.sig_eta_entropy = []
        self.sig_eta_pf_entropy = []
        self.x_prior_entropy = []
        self.x_prior_neg_entropy = []
        self.x_entropy = []
        self.propagate_entropy = []
        self.zg_prob = []
        self.kls = []
        self.prior_posterior_kl = []
        self.propagate_kl = []
        self.kl_terms = []
        self.z3_ms = []
        self.z3_sigs = []
        self.l2s = []
        self.risk = []
        self.em_iter = 0  # count
        self.tt = 0

    def save_traj(self, res_dir):
        x = np.asarray([c.mu_x0_m for c in self.cells])
        u = np.asarray([c.mu_u0_m for c in self.cells])
        z = np.asarray([c.mu_z0_m for c in self.cells])
        xu = np.hstack((x, u))
        np.save(os.path.join(res_dir, "xu_plan.npy"), xu)
        np.save(os.path.join(res_dir, "x_plan.npy"), x)
        np.save(os.path.join(res_dir, "u_plan.npy"), u)
        np.save(os.path.join(res_dir, "z_plan.npy"), z)

    def converged(self):
        delta_tol_pcnt = 0.005
        if len(self.costs_m) > 2:
            delta_pcnt = abs(self.costs_m[-1] - self.costs_m[-2]) / self.costs_m[-1]
            return delta_pcnt < delta_tol_pcnt
        else:
            return False

    def save(self, path, name):
        filename = f"i2c_{name}.pkl"
        with open(os.path.join(path, filename), "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = dill.load(f)
        return obj

    #############################
    # Plotting code
    #############################
    def plot_traj(self, title="", dir_name=None, filename="traj"):
        _opacity = 0.3
        t = range(self.H)

        mu_xu_f, sig_xu_f = self.get_prior_state_action_distribution()
        mu_xu_m, sig_xu_m = self.get_marginal_state_action_distribution()

        fig, a = plt.subplots(self.sys.dim_xu, 1)

        a[0].set_title(f"{self.sys.name} {title}")

        for i in range(self.sys.dim_xu):
            name = self.sys.key[i]
            unit = self.sys.unit[i]
            g = (
                self.sys.zgc[i] * np.ones((self.H,))
                if not np.any(np.equal(self.sys.zgc[i], None))
                else None
            )

            xf_u, xf_l = self.indexed_confidence_bound(mu_xu_f, sig_xu_f, i)
            xm_u, xm_l = self.indexed_confidence_bound(mu_xu_m, sig_xu_m, i)

            a[i].fill_between(
                t, xf_l, xf_u, where=xf_u >= xf_l, facecolor="m", alpha=_opacity
            )
            a[i].fill_between(
                t, xm_l, xm_u, where=xm_u >= xm_l, facecolor="c", alpha=_opacity
            )

            a[i].plot(t, mu_xu_f[:, i], "m.-", label="Prior")
            a[i].plot(t, mu_xu_m[:, i], "c.-", label="Posterior")

            if g is not None:
                a[i].plot(t, g, "k--", label="Goal")

            a[i].set_ylabel(f"{name} ({unit})" if unit is not None else name)
            a[i].legend(loc="upper left")
            if g is not None:
                a[i].plot(self.tau, g[self.tau], "k*", markersize=10)

        a[-1].set_xlabel("Timesteps")
        if dir_name is not None:
            plt.savefig(
                os.path.join(dir_name, f"{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            if PLOT_TIKZ:
                tikzplotlib.save(os.path.join(dir_name, f"{filename}.tex"))
            plt.close(fig)
        return fig

    def plot_propagate(self, dir_name=None, filename=""):
        fig, a = plt.subplots(self.sys.dim_xu, 1)
        t = range(self.H)
        _opacity = 0.3
        mu_xu0_m, sig_xu0_m = self.get_marginal_state_action_distribution()
        mu_xu0_pf, sig_xu0_pf = self.get_propagated_state_action()

        for i in range(self.sys.dim_xu):
            name = self.sys.key[i]
            unit = self.sys.unit[i]

            xm_u, xm_l = self.indexed_confidence_bound(mu_xu0_m, sig_xu0_m, i)
            xpf_u, xpf_l = self.indexed_confidence_bound(mu_xu0_pf, sig_xu0_pf, i)
            a[i].fill_between(
                t, xm_u, xm_l, where=xm_u >= xm_l, facecolor="b", alpha=_opacity
            )

            a[i].fill_between(
                t, xpf_u, xpf_l, where=xpf_u >= xpf_l, facecolor="r", alpha=_opacity
            )

            a[i].plot(t, xm_u, "b--")
            a[i].plot(t, xm_l, "b--")
            a[i].plot(t, xpf_u, "r--")
            a[i].plot(t, xpf_l, "r--")
            a[i].plot(t, mu_xu0_pf[:, i], "rx-", label="Forward")
            a[i].plot(t, mu_xu0_m[:, i], "b.-", label="Marginal")

            a[i].set_ylabel(f"{name} ({unit})" if unit is not None else name)
            a[i].legend(loc="upper left")

        if dir_name is not None:
            plt.savefig(
                os.path.join(dir_name, f"propagate_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            if PLOT_TIKZ:
                tikzplotlib.save(os.path.join(dir_name, f"propagate_{filename}.tex"))
            plt.close(fig)
        return fig

    def plot_alphas(self, dir_name=None, filename=""):
        f = plt.figure()
        plt.plot(self.alphas, "o-", label="actual")
        plt.plot(self.alphas_desired, "o-", label="desired")
        plt.plot(self.alpha_risk, "o-", label="EVaR")
        plt.plot(self.alphas_pf, "o-", label="Propagated")
        plt.grid(True)
        plt.legend()
        plt.title("Sigma Xi scale factor over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("1 / Alpha")
        if dir_name:
            plt.savefig(
                os.path.join(dir_name, f"alpha_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)


    def plot_cost(self, dir_name, filename):

        f, axes = plt.subplots(1, 1)
        a = axes
        _a = a.twinx()
        a.grid()
        a.set_title("Quadratic Cost and LogLikelihood over Message Iterations")

        cost_m_mu = np.asarray(self.costs_m)
        cost_m_sigma = np.asarray(self.costs_m_var)
        cost_m_2std = 2 * np.sqrt(cost_m_sigma)
        cost_m_upper = cost_m_mu + cost_m_2std
        cost_m_lower = cost_m_mu - cost_m_2std
        iters = np.arange(0, cost_m_mu.shape[0])
        if self._propagate:
            cost_pf_mu = np.asarray(self.costs_pf)
            cost_pf_sigma = np.asarray(self.costs_pf_var)
            cost_pf_2std = 2 * np.sqrt(cost_pf_sigma)
            cost_pf_upper = cost_pf_mu + cost_pf_2std
            cost_pf_lower = cost_pf_mu - cost_pf_2std
            a.fill_between(
                iters,
                cost_pf_lower,
                cost_pf_upper,
                where=cost_pf_lower <= cost_pf_upper,
                color="r",
                alpha=0.3,
            )
            a.plot(self.costs_pf, "r", label="Propagated")
        a.fill_between(
            iters,
            cost_m_lower,
            cost_m_upper,
            where=cost_m_lower <= cost_m_upper,
            color="b",
            alpha=0.3,
        )
        a.plot(self.costs_m, "b", label="Marginal")

        if dir_name:
            plt.savefig(
                os.path.join(dir_name, f"msg_cost_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def plot_cost_all(self, dir_name, filename):
        f, axes = plt.subplots(3, 1)
        a = axes[0]
        _a = a.twinx()
        a.set_title("Quadratic Cost and LogLikelihood over Iterations")
        a.plot(self.costs_p_all, "ro-", label="Predictive")
        a.plot(self.costs_m_all, "rx-", label="Marginal")
        a.plot(self.costs_pf_all, "m^-", label="Propagated")

        a.set_ylabel("Cost", color="r")
        a.legend(loc="lower left")
        a.grid()
        _a.plot(self.likelihoods_all, "b.-", label="Message Iter Full")
        _a.grid()
        _a.legend(loc="lower right")
        _a.set_ylabel("Log Likelikelihood", color="b")

        _a = axes[1].twinx()
        _a.plot(self.likelihoods_xu_all, "co-", label="Message Iter State Action")
        _a.grid()
        _a.legend(loc="lower right")
        _a.set_ylabel("Log Likelikelihood", color="b")

        axes[-1].set_xlabel("EM Iterations")
        if dir_name:
            plt.savefig(
                os.path.join(dir_name, f"msg_cost_all_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def plot_observed_traj(self, filename="", dir_name=None):
        mu_z_m = np.asarray([c.mu_z0_m for c in self.cells]).reshape(
            (-1, self.sys.dim_z)
        )
        cov_z_m = np.asarray([c.sig_z0_m for c in self.cells]).reshape(
            (-1, self.sys.dim_z, self.sys.dim_z)
        )
        t = range(self.H)

        f, _a = plt.subplots(self.sys.dim_z, 1)
        _a = [_a] if self.sys.dim_z == 1 else _a
        for i in range(self.sys.dim_z):
            z_u, z_l = self.indexed_confidence_bound(mu_z_m, cov_z_m, i)
            g = self.sys.zg[i] * np.ones((self.H,))
            _a[i].set_ylabel(self.sys.z_key[i])
            _a[i].fill_between(t, z_l, z_u, where=z_u >= z_l, facecolor="b", alpha=0.3)
            _a[i].plot(mu_z_m[:, i])
            _a[i].plot(g, "k--")
        _a[0].set_title("Observation")
        if dir_name is not None:
            plt.savefig(
                os.path.join(dir_name, f"observation_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def plot_terminal_observed_traj(self, filename="", dir_name=None):
        dim_z_term = self.Qf.shape[0]  # TODO fix
        z_m = np.asarray(self.z3_ms)[:, :dim_z_term].reshape((-1, dim_z_term))
        z_sig = np.asarray(self.z3_sigs)[:, :dim_z_term, :dim_z_term]
        H = z_m.shape[0]
        f, _a = plt.subplots(dim_z_term, 1)
        for i in range(dim_z_term):
            zg = self.sys.zg_term[i]
            zg_std = np.sqrt(self.sig_xi_terminal[i, i])
            g_upper = (zg + 2 * zg_std) * np.ones((H,))
            g_lower = (zg - 2 * zg_std) * np.ones((H,))
            z = z_m[:, i]
            z_std = np.sqrt(z_sig[:, i, i])
            z_upper = z + 2 * z_std
            z_lower = z - 2 * z_std
            g = zg * np.ones((H,))
            _a[i].plot(z, "b-")
            _a[i].plot(z_upper, "b--")
            _a[i].plot(z_lower, "b--")
            _a[i].plot(g, "k-")
            _a[i].plot(g_upper, "k--")
            _a[i].plot(g_lower, "k--")
        _a[0].set_title("Terminal Observation")
        if dir_name is not None:
            plt.savefig(
                os.path.join(dir_name, f"terminal_observation_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def plot_system_dynamics(self, filename="", dir_name=None):
        Aeig_sys = np.asarray(
            [np.sort(np.linalg.eig(c.A)[0]) for c in self.cells]
        ).squeeze()
        Aeig_sys_m = np.absolute(Aeig_sys)
        Aeig_sys_p = np.angle(Aeig_sys)
        Bs = np.asarray([c.B for c in self.cells])
        _a = np.asarray([c.a for c in self.cells])
        E = np.asarray([c.E for c in self.cells])
        f, a = plt.subplots(5, 1)
        a[0].set_title("Linearized Dynamics")
        a[0].plot(Aeig_sys_m)
        a[0].set_ylabel("A Eigenvalue Magnitude")
        a[1].plot(Aeig_sys_p)
        a[1].set_ylabel("A Eigenvalue Phase")
        a[2].set_ylabel("B Elements")
        for i in range(self.sys.dim_x):
            for j in range(self.sys.dim_u):
                a[2].plot(Bs[:, i, j], label=f"B{i}{j}")
        a[2].legend()
        for i in range(self.sys.dim_x):
            a[3].plot(_a[:, i], label=f"a{i}")
        a[3].set_ylabel("a Elements")
        a[3].legend()
        for j in range(self.sys.dim_x):
            for i in range(self.sys.dim_z):
                a[4].plot(E[:, i, j], label=f"E{i}{j}")
        a[4].set_ylabel("E Elements")
        a[4].legend(ncol=self.sys.dim_x)
        a[-1].set_xlabel("Timesteps")
        if dir_name is not None:
            plt.savefig(
                os.path.join(dir_name, f"system_dynamics_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def plot_controller(self, filename="", dir_name=None, lqr_compare=False):
        t = range(self.H)
        K, k, _ = self.get_local_linear_policy()
        u_pol = np.asarray([c.u_pol for c in self.cells])
        u_pol_K = np.asarray([c.u_pol_K for c in self.cells])
        u_pol_k = np.asarray([c.k for c in self.cells])
        um_m = np.asarray([c.mu_u0_m for c in self.cells])

        if lqr_compare:
            _, u_lqr, K_lqr, k_lqr = self.finite_horizon_lqr()

        f, a = plt.subplots(self.sys.dim_u + 2, 1)
        plt.title("Linear Gaussian Controllers")
        idx = 0
        for i in range(self.sys.dim_u):
            a[i].plot(t, um_m[:, i], "k.-", label="marginal")
            a[i].plot(t, u_pol_K[:, i], "g.-", label="Feedback")
            a[i].plot(t, u_pol_k[:, i], "b.-", label="Feedforward")
            a[i].plot(t, u_pol[:, i], "r.", label="Controller")
            if lqr_compare:
                a[i].plot(t, u_lqr[:, i], "y.", label="LQR")
            a[i].legend()
            idx += 1

        for i in range(self.sys.dim_u):
            for j in range(self.sys.dim_x):
                a[idx].plot(t, K[:, i, j], label=f"K{i}{j}")
                if lqr_compare:
                    a[idx].plot(t, K_lqr[:, i, j], label=f"K{i}{j}_lqr")
        a[idx].legend()
        idx += 1
        for i in range(self.sys.dim_u):
            a[idx].plot(t, k[:, i], label=f"k{i}")
            if lqr_compare:
                a[idx].plot(t, k_lqr[:, i], label=f"k{i}_lqr")
        a[idx].legend()

        if dir_name is not None:
            plt.savefig(
                os.path.join(dir_name, f"controller_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def plot_ricatti(self, filename="", dir_name=None):
        lam_x3 = np.asarray([c.lambda_x3_b for c in self.cells]).squeeze() * self.alpha
        nu_x3 = np.asarray([c.nu_x3_b for c in self.cells]).squeeze() * self.alpha
        f, a = plt.subplots(2, 1)
        a[0].set_title("lam x3 b")
        for i in range(self.sys.dim_x):
            for j in range(self.sys.dim_x):
                a[0].plot(lam_x3[:, i, j], label=f"lambda x b {i}{j}")
        a[0].legend()
        a[1].set_title("nu x3 b")
        for i in range(self.sys.dim_x):
            a[1].plot(nu_x3[:, i], label=f"nu x b {i}")
        a[1].legend()
        if dir_name is not None:
            plt.savefig(
                os.path.join(dir_name, f"ricatti_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def plot_uncertainty(self, filename="", dir_name=None):
        t = range(self.H)
        sig_eta = np.asarray([c.sig_eta for c in self.cells]).squeeze()
        sig_eta_pf = np.asarray([c.sig_eta_pf for c in self.cells]).squeeze()

        sig_eta_sys = np.repeat((self.sys.sig_eta,), self.H, axis=0)

        sigx1 = np.asarray([c.sig_x1_f for c in self.cells]).squeeze()

        sig_x0_m = np.asarray([c.sig_x0_m for c in self.cells]).squeeze()
        sigu0_f = np.asarray([c.sig_u0_f for c in self.cells])
        sigu1_f = np.asarray([c.sig_u1_f for c in self.cells])
        sigu0_m = np.asarray([c.sig_u0_m for c in self.cells])

        f, a = plt.subplots(self.sys.dim_x + self.sys.dim_u, 1)
        plt.title("Uncertainties")
        for i in range(self.sys.dim_x):
            a[i].plot(t, sigx1[:, i, i], "b.-", label="sig x1 fwd")
            a[i].plot(t, sig_x0_m[:, i, i], "g.-", label="sig x0 mrg")
            a[i].legend(loc="lower left")
            a[i].set_ylabel("SigX1", color="b")
            _a = a[i].twinx()
            _a.plot(t, sig_eta[:, i, i], "r.-", label="sig eta")
            _a.plot(t, sig_eta_pf[:, i, i], "c.-", label="sig eta pf")
            _a.set_ylabel("sig eta", color="r")
            _a.plot(t, sig_eta_sys[:, i, i], "m--", label="sig eta sys")

            _a.legend(loc="lower right")
        for i in range(self.sys.dim_u):
            j = self.sys.dim_x + i
            a[j].plot(t, sigu0_f[:, i, i], "c.-", label="prior")
            a[j].plot(t, sigu1_f[:, i, i], "m.-", label="filtered")
            a[j].plot(t, sigu0_m[:, i, i], "g.-", label="marginal")
            a[j].set_ylabel("SigU")
            a[j].legend()
        if dir_name is not None:
            plt.savefig(
                os.path.join(dir_name, f"uncertainties_{filename}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def plot_metrics(self, episode, index, dir_name=None, filename=""):
        """Configurable selection of metrics of choice"""

        iter_name = f"{filename}_iter_{episode}_{index}"

        self.plot_traj(f"iteration {index}", dir_name=dir_name, filename=iter_name)
        self.plot_uncertainty(dir_name=dir_name, filename=iter_name)
        self.plot_controller(dir_name=dir_name, filename=iter_name)
        if self._propagate:
            self.plot_propagate(dir_name=dir_name, filename=iter_name)
        if index > 0:  # not interesting at the start
            ep_name = f"{episode}_{filename}"
            self.plot_cost(dir_name, ep_name)
            self.plot_alphas(dir_name=dir_name, filename=filename)
