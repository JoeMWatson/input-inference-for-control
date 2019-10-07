"""
Implementation of Input Inference for Control (i2c)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from copy import deepcopy
from contextlib import contextmanager
import pdb
import os
import matplotlib2tikz
import dill

import cProfile

from pi2c.utils import converged_list, finite_horizon_lqr


PLOT_PDF = False
# PLOT_TIKZ = True
PLOT_TIKZ = False
# run the model in pure feedforward during inference- not all that helpful?
# use simulator to tune priors instead
SIMULATE = False

CHECK_COVAR = False # check the are semi pos def, NOTE really slows things down

DEBUG_PLOTS = False

def moment2information(mu, sigma):
    # TODO cholesky?
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


class I2cCell(object):
    """A single time index of linear gaussian i2c, like Fig 1. in the paper."""


    def __init__(self, i, sys, sigU, sigXi, lamXi, bkwd_sf):
        self.index = i
        self.sys = sys
        self.sg = np.copy(self.sys.sg)
        self.sigU = sigU
        self.sigXi = sigXi
        self.lamXi = lamXi  # often used more than sigXi
        self.mu_u0_base = np.zeros((self.sys.dim_u, 1))
        # random dither speeds up optimization but makes loss curve non-decreasing
        # and is hacky!
        # self.mu_u0_base = 1e-6 * np.random.randn(self.sys.dim_u, 1)
        self.sig_u0_base = sigU
        self.mu_u0_f = np.copy(self.mu_u0_base)
        self.mu_u1_f = np.copy(self.mu_u0_base) # used for linearization below
        self.sig_u0_f = np.copy(self.sig_u0_base)
        self.mu_u0_prev = np.copy(self.mu_u0_f)
        self.sig_u0_prev = np.copy(self.sig_u0_f)
        self.sig_u_base = 0.1 * np.eye(1)
        self.mu_x0_f = np.copy(self.sys.x0)
        self.mu_x0_f_prev = np.copy(self.sys.x0)
        self.bkwd_sf = bkwd_sf
        self.mu_y_f = np.copy(self.sys.x0)
        self.sigma_y = np.diag([0.05, 0.05, 0.1, 0.1])
        # locally linear stochastic system params
        self.dyn_linearized = False
        self.obs_linearized = False
        self.mu_x_lin = None
        self.mu_u_lin = None
        self.sig_x_lin =  5e-4 * np.eye(self.sys.dim_x)
        self.sig_u_lin = 5e-4 * np.eye(self.sys.dim_u)
        self.lam_x_lin = np.linalg.inv(self.sig_x_lin)
        self.lam_u_lin = np.linalg.inv(self.sig_u_lin)
        self.A = None
        self.B = None
        self.a = None
        self.sigEta = None
        self.E = None
        self.e = None
        self.F = None

        self.mu_x3_m_prev = None
        self.sig_x3_m_prev = None

    def state_changed(self):
        # stop pointless relinearizations by checking state dist. change
        # speeds the initial stuff alot, but later on causes discontiuinities
        # - should do batch instead maybe rather than per timestep
        return True

    def _forward_kalman(self, mu0, sig0):
        self.mu_x0 = mu0
        self.sig_x0 = sig0
        self.xp = self.sys.A.dot(self.mu_x0)
        self.Vp = self.sys.A.dot(self.sig_x0.dot(self.sys.A.T)) + self.sys.sigEta + self.sys.B.dot(self.sigU.dot(self.sys.B.T))
        S = self.sigXi + self.sys.C.dot(self.Vp .dot(self.sys.C.T))
        K = self.Vp.dot(self.sys.C.T.dot(np.linalg.inv(S)))
        _IKC = np.eye(self.sys.dim_x ) - K.dot(self.sys.C)
        self.xf = self.xp + K.dot(self.sg - self.sys.C.dot(self.xp))
        self.Vf = _IKC.dot(self.Vp.dot(_IKC.T)) + K.dot(self.sigXi.dot(K.T))
        self.mu_x2 = self.xp
        self.sig_x2 = self.Vp
        self.mu_x3 = self.xf
        self.sig_x3 = self.Vf
        return self.xf, self.Vf

    def _simulate(self, mu0):
        self.mu_x0_s = mu0
        self.mu_x3_s = self.sys(self.mu_x0_s, self.mu_u0_f)[0]
        return self.mu_x3_s

    def _forward_msgs(self, mu0, sig0):
        self.mu_x0_f = mu0
        self.sig_x0_f = sig0
        assert is_pos_def(self.sig_x0_f)

        state_changed = self.state_changed()

        # innovate state
        self.nu_x0_f, self.lambda_x0_f = moment2information(
            self.mu_x0_f, self.sig_x0_f)

        # linearize observation model about prior
        if self.obs_linearized and not state_changed:
            self.mu_z0_f = self.E.dot(self.mu_x0_f) + self.F.dot(self.mu_u0_f) + self.e
        else:
            self.mu_z0_f, self.E, self.e, self.F = self.sys.observe(
                self.mu_x0_f, self.mu_u0_f)
            self.obs_linearized = True

        # required for backwards computation
        self.sig_z1_f = self.sigXi + self.F.dot(self.sig_u0_f.dot(self.F.T))
        self.lambda_z1_f = np.linalg.inv(self.sig_z1_f)
        self.nu_z1_f = self.E.T.dot(self.lambda_z1_f.dot(
             self.sys.sg - self.F.dot(self.mu_u0_f) - self.e))
        self.nu_x1_f = self.nu_x0_f + self.nu_z1_f
        self.lambda_x1_f = self.lambda_x0_f + self.E.T.dot(
            self.lambda_z1_f.dot(self.E))

        self.mu_x1_f, self.sig_x1_f = information2moment(
            self.nu_x1_f, self.lambda_x1_f)

        # init action (done in __init__ and _maximize)
        # innovate action
        self.nu_u0_f, self.lambda_u0_f = moment2information(
            self.mu_u0_f, self.sig_u0_f)
        self.sig_z2_f = self.sigXi + self.E.dot(self.sig_x0_f.dot(self.E.T))
        self.lambda_z2_f = np.linalg.inv(self.sig_z2_f)
        self.nu_z2_f = self.F.T.dot(self.lambda_z2_f.dot(
           self.sys.sg -self.E.dot(self.mu_x0_f) -self.e))
        self.lambda_u1_f = self.lambda_u0_f + self.F.T.dot(
            self.lambda_z2_f.dot(self.F))
        self.nu_u1_f = self.nu_u0_f + self.nu_z2_f

        self.mu_u1_f, self.sig_u1_f = information2moment(
            self.nu_u1_f, self.lambda_u1_f)
        assert is_pos_def(self.sig_u1_f)

        # propagate state and action mean, get linearizations from priors in first pass, use marginals for subsequent passes
        if self.dyn_linearized and not state_changed:
            self.mu_x3_f = self.A.dot(self.mu_x1_f) + self.B.dot(self.mu_u1_f) + self.a
        else:
            self.mu_x3_f, self.A, self.a, self.B, self.sigEta = self.sys(self.mu_x1_f, self.mu_u1_f)
            self.dyn_linearized = True
        assert is_pos_def(self.sigEta)

        # propagate action uncertainty
        self.sig_u2_f = self.B.dot(self.sig_u1_f.dot(self.B.T))

        # propagate state uncertainty
        self.sig_x2_f = self.A.dot(self.sig_x1_f.dot(self.A.T)) + self.sigEta
        self.sig_x3_f = self.sig_x2_f + self.sig_u2_f
        assert is_pos_def(self.sig_x2_f)
        self.lambda_x2_f = np.linalg.inv(self.sig_x2_f) # needed in backward pass
        self.nu_x3_f, self.lambda_x3_f = moment2information(
            self.mu_x3_f, self.sig_x3_f)

        self.mu_x0_f_prev = np.copy(self.mu_x0_f)

        return self.mu_x3_f, self.sig_x3_f

    def _backward_kalman(self, mu_end, sig_end):
        if mu_end is None and sig_end is None:
            self.mu_x3_b = self.mu_x3
            self.sig_x3_b = self.sig_x3
        else:
            self.mu_x3_b = mu_end
            self.sig_x3_b = sig_end
        J = self.sig_x0.dot(self.sys.A.T.dot(np.linalg.inv(self.sig_x2)))
        self.xs = self.mu_x0 + J.dot(self.mu_x3_b - self.sys.A.dot(self.mu_x0))
        self.Vs = self.sig_x0 + J.dot((self.sig_x3_b - self.sig_x2).dot(J.T))
        self.mu_x0_b = self.xs
        self.sig_x0_b = self.Vs
        return self.xs, self.Vs

    def _backward_msgs(self, mu_end, sig_end):
        end_in_chain = mu_end is None and sig_end is None
        if end_in_chain:
            if self.bkwd_sf is not None:
                self.mu_x3_m = self.mu_x3_f
                self.sig_x3_m = self.bkwd_sf * self.sig_x3_f
            else:
                # for LQR, we want lax_x3_b = lamXi and nu_x3_b = equivalent
                # lamXi_last = 20 * self.lamXi
                lamXi_last = 1 * self.lamXi
                self.lambda_x3_b =self.E.T.dot(lamXi_last.dot(self.E))
                self.sig_x3_m = np.linalg.inv(
                    self.lambda_x3_f + self.lambda_x3_b)
                self.nu_x3_b = self.E.T.dot(lamXi_last.dot(self.sys.sg - self.e))
                self.mu_x3_m = self.sig_x3_m.dot(
                    self.nu_x3_f + self.nu_x3_b) # so nu_x3_b = Qxg
        else:
            self.mu_x3_m = mu_end
            self.sig_x3_m = sig_end

        assert is_pos_def(self.sig_x3_m), "{} {}".format(self.index, np.linalg.eigvals(self.sig_x3_m))
        assert is_pos_def(self.lambda_x2_f), "{} {}".format(self.index, np.linalg.eigvals(self.lambda_x2_f))

        # shortcut state de-innovation via marginal equality and get auxillary after
        self.lambda_x2_a = self.lambda_x3_f - self.lambda_x3_f.dot(self.sig_x3_m.dot(self.lambda_x3_f))
        assert is_pos_def(self.lambda_x2_a), "{} {}".format(self.index, np.linalg.eigvals(self.lambda_x2_a))
        self.nu_x2_a = self.nu_x3_f - self.lambda_x3_f.dot(self.mu_x3_m)
        # shortcut action for now cos of auxillary
        self.lambda_x1_a = self.A.T.dot(self.lambda_x2_a.dot(self.A))
        assert is_pos_def(self.lambda_x1_a), "{} {}".format(self.index, np.linalg.eigvals(self.lambda_x1_a))
        self.nu_x1_a = self.A.T.dot(self.nu_x2_a)
        # marginalize
        self.sig_x0_m = self.sig_x1_f - self.sig_x1_f.dot(self.lambda_x1_a.dot(self.sig_x1_f))
        self.mu_x0_m = self.mu_x1_f - self.sig_x1_f.dot(self.nu_x1_a)

        assert is_pos_def(self.sig_x0_m), "{} {}".format(self.index, np.linalg.eigvals(self.sig_x0_m))

        # get action from auxillary
        self.lambda_u2_a = self.lambda_x2_a
        self.nu_u2_a = self.nu_x2_a
        # depropagate action
        self.lambda_u1_a = self.B.T.dot(self.lambda_u2_a.dot(self.B))
        self.nu_u1_a = self.B.T.dot(self.nu_u2_a)
        # marginalize action, which then passes through equality
        self.sig_u0_m = self.sig_u1_f - self.sig_u1_f.dot(self.lambda_u1_a.dot(self.sig_u1_f))
        assert is_pos_def(self.sig_u0_m)
        self.mu_u0_m = self.mu_u1_f - self.sig_u1_f.dot(self.nu_u1_a)
        if np.isnan(self.mu_u0_m).any():
            print("ERROR nan in self.mu_u0_m")
            print(self.mu_u1_f, self.sig_u1_f, self.nu_u1_a)

        # get marginalised observation
        z, C, _, D = self.sys.observe(self.mu_x0_m, self.mu_u0_m)
        self.mu_z0_m = z
        self.sig_z0_m = C.dot(self.sig_x0_m.dot(C.T)) + D.dot(self.sig_u0_m.dot(D.T))

        return self.mu_x0_m, self.sig_x0_m

    def _backward_ricatti_msgs(self, nu_b, lambda_b):
        end_in_chain = nu_b is None and lambda_b is None
        if end_in_chain:
            if self.bkwd_sf is None: # inversion precision issues messes with LQR comparison
                # handled in backward pass
                pass
            else:
                self.nu_x3_b = np.linalg.solve(self.sig_x3_m, self.mu_x3_m) - self.nu_x3_f
                self.lambda_x3_b = np.linalg.inv(self.sig_x3_m) - self.lambda_x3_f
        else:
            self.nu_x3_b = nu_b
            self.lambda_x3_b = lambda_b

        # backwards ricatti equation
        # note: maybe in the future we get the backwards from the forwards and
        # marginal but for now i wanna make sure the maths is correct
        Q = self.E.T.dot(self.lambda_z1_f.dot(self.E))
        R = self.F.T.dot(self.lambda_z2_f.dot(self.F))
        Rug = self.nu_z2_f
        nu_u_0 = np.linalg.solve(self.sig_u0_f, self.mu_u0_f)
        gamma = np.dot(self.lambda_x2_f,
            np.linalg.inv(self.lambda_x2_f + self.lambda_x3_b))

        # precision
        ALA = self.A.T.dot(self.lambda_x3_b.dot(self.A))
        M = np.linalg.inv(self.sigEta + self.sig_u2_f) + self.lambda_x3_b
        ALMLA = self.A.T.dot(
            self.lambda_x3_b.dot(
                np.linalg.solve(M, self.lambda_x3_b.dot(self.A))))
        self.lambda_x0_b = Q + ALA - ALMLA

        # mean
        AILM = self.A.T.dot(np.eye(self.sys.dim_x) -
                            self.lambda_x3_b.dot(np.linalg.inv(M)))

        self.nu_x0_b = self.nu_z1_f + AILM.dot(
            self.nu_x3_b - self.lambda_x3_b.dot(self.a)
            - self.lambda_x3_b.dot(self.B.dot(self.mu_u1_f)))

        gamma_L = gamma.dot(self.lambda_x3_b)
        igamma = np.eye(self.sys.dim_x) - gamma
        sig_x3_b = np.linalg.inv(self.lambda_x3_b)
        self.lambda_x2_b = np.linalg.inv(sig_x3_b + self.sig_u2_f)
        self.mu_u2_f = self.B.dot(self.mu_u1_f)
        self.nu_x2_b = self.lambda_x2_b.dot(sig_x3_b.dot(self.nu_x3_b) - self.mu_u2_f)

        psi = gamma_L.dot(self.sig_x2_f.dot(
            self.lambda_x2_f + np.linalg.inv(sig_x3_b + self.sig_u2_f)))

        sig_u = self.sig_u0_m

        K = -sig_u.dot(self.B.T.dot(
            psi.dot(self.A)))

        k = sig_u.dot(nu_u_0 + Rug + self.B.T.dot(
            gamma.dot(self.nu_x3_b)
            +igamma.dot(self.nu_x2_b)
            - psi.dot(self.a)))

        self.u_pol = K.dot(self.mu_x0_m) + k
        self.u_pol_K = K.dot(self.mu_x0_m)
        self.K = K
        self.k = k
        self.sigK = self.sig_u0_m

        return self.nu_x0_b, self.lambda_x0_b

    def linearize(self):
        """Relinearize about marginals"""
        _, A, a, B, self.sigEta = self.sys(self.mu_x0_m, self.mu_u0_m)
        _, E, e, F = self.sys.observe(self.mu_x0_m, self.mu_u0_m,)
        assert is_pos_def(self.sigEta)
        if self.are_nan(A, a, B, E, e, F):
            print("Error NANs in linearization!")
            print(A, a, B, E, e, F)
            self.linearized = False
        else:
            self.A, self.a, self.B, self.E, self.e, self.F = A, a, B, E, e, F
            self.linearized = True

    def linearize_prior(self):
        """Relinearize about prior"""
        x, A, a, B, sigEta = self.sys(self.mu_x0_f, self.mu_u1_f)
        y, E, e, E = self.sys.observe(self.mu_x0_f, self.mu_u1_f)
        self.sigEta = sigEta

        if self.are_nan(A, a, B, E, e, F):
            print("Error NANs in linearization!")
            print(A, a, B, E, e, F)
            self.linearized = False
        else:
            self.A, self.a, self.B, self.E, self.e, self.F = A, a, B, E, e, F
            self.linearized = True

    @staticmethod
    def are_nan(*args):
        are_nan = False
        for a in args:
            are_nan = are_nan or np.any(np.isnan(a))
        return are_nan


class I2cGraph(object):
    """Manages i2c for a whole trajectory"""

    def __init__(self, sys, horizon, Q, R, alpha, alpha_update_tol, sigU,
                 msg_iter=100, msg_tol=1e-3, ll_tol=1e-3, bkwd_sf=1.0, res_dir=None):
        # NOTE: the alpha here is 1/alpha in the paper! it is easier to reason about
        self.sys = sys # dynamics model
        self.H = horizon
        self.sg = np.copy(self.sys.sg)
        self.alpha_base = alpha
        self.alpha = self.alpha_base
        self.alphas = [self.alpha]
        self.alphas_desired = [self.alpha]
        self.alpha_update_tol = alpha_update_tol
        self.alpha_tv = self.alpha * np.ones((self.H,)) # time-varying is bad idea?
        self.alphas_tv = [np.copy(self.alpha_tv)]
        self.Q = Q
        self.R = R
        # TODO use np.block
        self.QR = np.zeros((self.sys.dim_y, self.sys.dim_y)) # \Theta in paper
        self.QR[:self.sys.dim_xa, :self.sys.dim_xa] = Q
        self.QR[self.sys.dim_xa:self.sys.dim_xa + self.sys.dim_u,
                self.sys.dim_xa:self.sys.dim_xa + self.sys.dim_u] = R
        self.lamXi0 = np.copy(self.QR)
        self.sigXi0 = np.linalg.inv(self.QR)
        self.bkwd_sf = bkwd_sf
        det_sigXi0 = np.linalg.det(self.sigXi0)
        assert det_sigXi0 > 0.0, "|{}| = {}".format(self.sigXi0, det_sigXi0)

        lamXi = np.linalg.inv(self.sigXi)
        self.cells = [I2cCell(i, sys, sigU, self.sigXi, lamXi, bkwd_sf)
                      for i in range(self.H)]
        self.reset_metrics(False)
        self.costs_m_all = []
        self.costs_p_all = []
        self.cost_diff_norms = []
        self.likelihoods_all = []
        self.likelihoods_xu_all = []

        # EM and Message Passage parameters
        self.em_iter = 0 # count
        self.msg_iter = msg_iter
        self.msg_iters = []
        self.msg_tol = msg_tol
        self.ll_tol = ll_tol
        self.em_covergence_errors = 0

        # for saving results
        self.res_dir = res_dir

        self.policy_valid = False

    @property
    def sigXi(self):
        return self.alpha * self.sigXi0

    def _forward_kalman(self):
        mu = np.copy(self.sys.x0)
        sig = np.copy(self.sys.sigX0)
        for c in self.cells:
            mu, sig = c._forward_kalman(mu, sig)

    def _backward_kalman(self):
        mu = None
        sig = None
        for c in reversed(self.cells):
            mu, sig = c._backward_kalman(mu, sig)

    def _simulate(self):
        mu = np.copy(self.sys.x0)
        for c in self.cells:
            mu = c._simulate(mu)

    def _forward_msgs(self):
        mu = np.copy(self.sys.x0)
        sig = np.copy(self.sys.sigX0)
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

    def _maximize(self):
        # aka update alpha
        # also has a time-varying alpha idea that was bad (but exciting)
        nan_traj = False
        s_covar = np.zeros((self.sys.dim_y, self.sys.dim_y))
        for i, c in enumerate(self.cells):
            if np.any(np.isnan(c.mu_z0_m)):
                print("y_m is nan")
                nan_traj = True
            else:
                err = c.mu_z0_m  - self.sg
                s_covar_t = (err.dot(err.T) + c.sig_z0_m)
                s_covar += s_covar_t
                tv = np.trace(np.linalg.solve(self.sigXi0, s_covar_t)) / float(self.sys.dim_y)
                self.alpha_tv[i] = np.clip(tv, 0.5, 50)
        s_covar = s_covar / float(self.H)
        s_covar = (s_covar + s_covar.T) / 2.0

        if nan_traj:
            self.plot_traj(
                -1, dir_name=self.res_dir,
                filename="nan_trajectory")

        alpha_update = np.trace(np.linalg.solve(self.sigXi0, s_covar)) / float(self.sys.dim_y)
        self.alphas_desired.append(alpha_update)
        if np.isnan(alpha_update):
            print("ERROR, alpha is nan, s_covar {}".format(s_covar))
        else:
            if not alpha_update > 0.0:
                print("S covar bad")
                print(s_covar)
                print(np.linalg.det(s_covar))
                print(np.linalg.eig(s_covar))
                print(self.sigXi0)
                self.plot_traj(
                    -1, dir_name=self.res_dir,
                    filename="bad_alpha")
                raise ValueError("{} <= 0.0".format(alpha_update))

            alpha_update_ratio = alpha_update / self.alpha
            self.alpha_update_tol_u = 2. - self.alpha_update_tol
            if alpha_update_ratio < self.alpha_update_tol:
                alpha_update = self.alpha_update_tol  * self.alpha
            if alpha_update_ratio > self.alpha_update_tol_u:
                alpha_update = self.alpha_update_tol_u * self.alpha


            self.alphas.append(alpha_update)
            self.alphas_tv.append(np.copy(self.alpha_tv))
            self.alpha = alpha_update
            lamXi = np.linalg.inv(self.sigXi)
            for c in self.cells:
                c.sigXi = self.sigXi
                c.lamXi = lamXi

        for c in self.cells:
            c.mu_u0_prev = c.mu_u0_f
            c.sig_u0_prev = c.sig_u0_f

        self.calc_cost()
        self.calc_gap()
        self.calc_likelihood()
        # record costs and likelihood here
        self.em_likelihoods.append((len(self.likelihoods), self.em_iter, self.likelihoods[-1]))
        self.em_cost.append(self.costs_m[-1])
        self.policy_entropy.append((self.em_iter, self.calc_policy_entropy()))

        # relinearize around marginal state (akin to system identification)
        # once you relinearize likelihood goes out the window
        # Note - doing this now in the forward pass, works better
        # as the forward is what we care about (closest to real system)
        # self._linearize()

    def calc_cost(self):
        cost_p = 0.0
        cost_m = 0.0
        if SIMULATE:
            cost_s = 0.0
        for c in self.cells:
            y_p = self.sys.observe(c.mu_x0_f, c.mu_u0_f)[0]
            y_m = self.sys.observe(c.mu_x0_m, c.mu_u0_m)[0]

            err_p = y_p - self.sg
            err_m = y_m - self.sg

            cost_p += err_p.T.dot(self.QR.dot(err_p))
            cost_m += err_m.T.dot(self.QR.dot(err_m))
            if SIMULATE:
                err_s = y_s - self.sg
                y_s = self.sys.observe(c.mu_x0_s, c.mu_u0_s)[0]
                cost_s += err_s.T.dot(self.QR.dot(err_s))
        self.costs_p.append(cost_p[0, 0])
        self.costs_m.append(cost_m[0, 0])
        if SIMULATE:
            self.costs_s.append(cost_s[0, 0])

    def calc_gap(self):
        x_f = np.asarray([c.mu_x0_f for c in self.cells]).squeeze()
        x_m = np.asarray([c.mu_x0_m for c in self.cells]).squeeze()
        gap = np.linalg.norm(x_m - x_f)
        self.traj_gaps.append(gap)

    def calc_policy_entropy(self):
        const = 2 * np.pi * np.e
        for i, c in enumerate(self.cells):
            det = np.linalg.det(c.sig_u0_m)
            if det <= 0.0:
                self.plot_traj(-1, dir_name=self.res_dir, filename="bad_sig_u")
                raise ValueError("{}, |{}| = {}".format(i, c.sig_u0_m, det))

        return sum([0.5 * np.log(np.linalg.det(const * c.sig_u0_m))
                    for c in self.cells])

    @staticmethod
    def covar(y_est, y):
        err = y - y_est
        return err.dot(err.T)

    def state_action_covar(self):
        covar = []
        for c in self.cells:
            J = c.sig_x0_f.dot(c.A.T.dot(np.linalg.inv(c.sig_x2_f)))  # TODO transpose and solve
            sig_x_lag = J.dot(c.sig_x3_m)
            x_00 = c.sig_x0_m + c.mu_x0_m.dot(c.mu_x0_m.T)
            x_10 = sig_x_lag + c.mu_x3_m.dot(c.mu_x0_m.T)
            x_11 = c.sig_x3_m + c.mu_x3_m.dot(c.mu_x3_m.T)
            x_covar = x_00 + x_10 + x_11
            covar.append(np.linalg.solve(c.sigEta, x_covar))
        return sum(covar)

    def _calc_likelihood(self):
        """Maths for likelihood."""
        ll_sig_w = -0.5 * self.H * np.linalg.det(self.sigXi)
        ll_sigv = -0.5 * sum([np.linalg.det(c.sigEta) for c in self.cells])
        ll_sigX0 = -0.5 * np.linalg.det(self.sys.sigX0)
        # this step depends on current A
        ll_mu_x0 = -0.5 * np.trace(np.linalg.solve(self.sys.sigX0,
            self.cells[0].sig_x2_f + self.covar(self.sys.x0, self.cells[0].mu_x0_m)))
        mu_z_covar = sum([self.covar(c.mu_z0_m, c.sg) for c in self.cells])
        sig_z_covar = sum([c.sig_z0_m for c in self.cells])
        ll_z = -0.5 * np.trace(np.linalg.solve(self.sigXi, mu_z_covar + sig_z_covar))
        ll_state = -0.5 * np.trace(self.state_action_covar())
        ll = ll_mu_x0 + ll_sigX0 + ll_sig_w + ll_sigv + ll_z + ll_state
        return ll, ll_state, ll_z

    def calc_likelihood(self):
        ll, ll_xu, ll_z = self._calc_likelihood()
        self.likelihoods.append(ll)
        self.likelihoods_xu.append(ll_xu)
        self.likelihoods_z.append(ll_z)

    def likelihood_xu_minima(self):
        if len(self.likelihoods_xu) > 3:
            going_up = self.likelihoods_xu[-1] > self.likelihoods_xu[-2]
            going_down = self.likelihoods_xu[-1] < self.likelihoods_xu[-2]
            went_down = self.likelihoods_xu[-2] < self.likelihoods_xu[-3]
            went_up = self.likelihoods_xu[-2] > self.likelihoods_xu[-3]
            minima = (going_down and went_up)# (going_up and went_down)# or
            if minima:
                print(self.likelihoods_xu[-3:])
            return minima
        else:
            return False

    def cost_minima(self):
        if len(self.costs_m) > 3:
            going_up = self.costs_m[-1] > self.costs_m[-2]
            went_down = self.costs_m[-2] < self.costs_m[-3]
            minima = going_up and went_down
            if minima:
                print(self.costs_m[-3:])
            return minima
        else:
            return False

    def finite_horizon_lqr(self):
        x_lqr, u_lqr, K, k, cost, Ps, ps = finite_horizon_lqr(
            self.H, self.sys.A, self.sys.a, self.sys.B,
            self.Q, self.R, self.sys.x0, self.sys.xg, np.zeros((self.sys.dim_u,)),
            self.sys.dim_x, self.sys.dim_u)
        return x_lqr, u_lqr, K, k

    def plot_traj(self, title, lqr_compare=False, plot_y=False, dir_name=None, filename="traj"):
        t = range(self.H)

        Vw = np.asarray([c.sigXi for c in self.cells]).squeeze()

        xp_m = np.asarray([c.mu_x3_f for c in self.cells]).squeeze()
        Vp_m = np.asarray([c.sig_x3_f for c in self.cells]).squeeze()
        xf_m = np.asarray([c.mu_x1_f for c in self.cells]).squeeze()
        Vf_m = np.asarray([c.sig_x1_f for c in self.cells]).squeeze()

        xm_m = np.asarray([c.mu_x0_m for c in self.cells]).squeeze()
        Vm_m = np.asarray([c.sig_x0_m for c in self.cells]).squeeze()
        um_m = np.asarray([c.mu_u0_m for c in self.cells]).reshape((self.H, self.sys.dim_u))
        Vum_m = np.asarray([c.sig_u0_m for c in self.cells])
        u0_m = np.asarray([c.mu_u0_prev for c in self.cells]).reshape((self.H, self.sys.dim_u))
        Vu0m = np.asarray([c.sig_u0_prev for c in self.cells])


        # simulate
        if SIMULATE:
            xs_m = np.asarray([c.mu_x0_s for c in self.cells]).squeeze()
            Vs_m = np.asarray([c.sig_x0_s for c in self.cells]).squeeze()

        if lqr_compare:
            x_lqr, u_lqr, K_lqr, k_lqr = self.finite_horizon_lqr()

        fig, a = plt.subplots(self.sys.dim_x + self.sys.dim_u, 1)
        a[0].set_title(title)

        for i in range(self.sys.dim_x):
            name = self.sys.key[i]
            unit = self.sys.unit[i]
            g = self.sys.sgc[i] * np.ones((self.H,))
            xp_l = xp_m[:, i] - 3. * Vp_m[:, i, i]
            xp_u = xp_m[:, i] + 3. * Vp_m[:, i, i]
            xf_l = xf_m[:, i] - 3. * Vf_m[:, i, i]
            xf_u = xf_m[:, i] + 3. * Vf_m[:, i, i]
            xm_l = xm_m[:, i] - 3. * Vm_m[:, i, i]
            xm_u = xm_m[:, i] + 3. * Vm_m[:, i, i]
            g_l = g - 3. * Vw[:, i, i]
            g_u = g + 3. * Vw[:, i, i]

            a[i].fill_between(t, xp_l, xp_u, where=xp_u>=xp_l, facecolor='y', alpha=0.4)
            a[i].fill_between(t, xf_l, xf_u, where=xf_u>=xf_l, facecolor='m', alpha=0.4)
            a[i].fill_between(t, xm_l, xm_u, where=xm_u>=xm_l, facecolor='c', alpha=0.4)

            a[i].plot(t, xp_m[:, i], 'y.-', label="Prediction")
            a[i].plot(t, xf_m[:, i], 'm.-', label="Filtered")
            a[i].plot(t, xm_m[:, i], 'c.-', label="Posterior")
            a[i].autoscale(axis='y')
            a[i].autoscale(False)
            if SIMULATE:
                a[i].plot(t, xs_m[:, i], 'r.', alpha=0.7, label="simulated")


            a[i].plot(t, g, 'k--', label="Goal")

            if lqr_compare:
                a[i].plot(t, x_lqr[:, i], 'rx', label="LQR")

            a[i].set_ylabel("{} ({})".format(name, unit)
                            if unit is not None else name)
            a[i].legend(loc="upper right")

        for i in range(self.sys.dim_u):
            j = self.sys.dim_x + i
            name = self.sys.key[j]
            unit = self.sys.unit[j]
            g = self.sg[j] * np.ones((self.H,))
            u0_l = u0_m[:, i] - 3. * Vu0m[:, i, i]
            u0_u = u0_m[:, i] + 3. * Vu0m[:, i, i]
            um_l = um_m[:, i] - 3. * Vum_m[:, i, i]
            um_u = um_m[:, i] + 3. * Vum_m[:, i, i]

            # plot twice so we can autoscale y without the priors dwarfing everything
            a[j].plot(t, u0_m[:, i], 'y.-')
            a[j].plot(t, um_m[:, i], 'c.-')

            a[j].autoscale(axis='y')
            a[j].autoscale(False)

            a[j].fill_between(t, u0_l, u0_u, where=u0_u>=u0_l, facecolor='y', alpha=0.4)
            a[j].fill_between(t, um_l, um_u, where=um_u>=um_l, facecolor='c', alpha=0.4)


            a[j].plot(t, u0_m[:, i], 'y.-', label=r"Prior")
            a[j].plot(t, um_m[:, i], 'c.-', label=r"Posterior")


            a[j].plot(t, g, 'k--', label="Goal")

            if lqr_compare:
                a[j].plot(t, u_lqr[:, i], 'rx', label="LQR")

            a[j].set_ylabel("{} ({})".format(name, unit)
                if unit is not None else name)
            a[j].legend(loc="lower right")

        a[-1].set_xlabel("Timesteps")
        if dir_name is not None:
            plt.savefig(os.path.join(dir_name, "{}.png".format(filename)),
                bbox_inches='tight', format='png')
            if PLOT_PDF:
                plt.savefig(os.path.join(dir_name, "{}.pdf".format(filename)),
                    bbox_inches='tight', format='pdf')
            if PLOT_TIKZ:
                matplotlib2tikz.save(
                    os.path.join(dir_name, "{}.tex".format(filename)))
            plt.close(fig)
        return fig

    def learn_kalman(self):
        self._forward_kalman()
        self._backward_kalman()

    @staticmethod
    def state_action_converged(state_action, state_action_prev, tol):
        if state_action_prev is not None:
            error = state_action - state_action_prev
            diff = abs(np.linalg.norm(error) / np.linalg.norm(state_action_prev))
            return diff < tol
        else:
            return False

    def get_state_action(self):
        x = np.asarray([c.mu_x0_m for c in self.cells])
        u = np.asarray([c.mu_u0_m for c in self.cells])
        return np.vstack((x, u))

    def get_state_and_action(self):
        x = np.asarray([c.mu_x0_m for c in self.cells])
        u = np.asarray([c.mu_u0_m for c in self.cells])
        return x,u

    def _linearize(self):
        for c in self.cells:
            c.linearize()

    def _linearize_prior(self):
        for c in self.cells:
            c.linearize_prior()

    def _update_means(self):
        for c in self.cells:
            c.mu_u0 = c.mu_u0_m

    def _update_priors(self):
        for c in self.cells:
            c.mu_u0_f = c.mu_u0_m
            c.sig_u0_f = c.sig_u0_m

    @staticmethod
    def abs_scale_factor_change(new, old):
        return abs(new - old) / abs(old + 1e-12)

    def inner_likelihood_converged(self, n):
        """."""
        if len(self.likelihoods) > 1:
            msg_delta = self.abs_scale_factor_change(
                self.likelihoods[-1], self.likelihoods[-2])
            if msg_delta < self.msg_tol:
                print("Inner converged after {}? delta {}->{} : {}".format(
                    n, self.likelihoods[-2], self.likelihoods[-1], msg_delta))
                return True
            else:
                return False
        else:
            return False

    def outer_likelihood_converged(self, n):
        """."""
        if len(self.outer_likelihoods) > 1:
            msg_delta = self.abs_scale_factor_change(
                self.outer_likelihoods[-1], self.outer_likelihoods[-2])
            if msg_delta < self.msg_tol:
                print("Outer converged after {}? delta {}->{} : {}".format(
                    n, self.outer_likelihoods[-2], self.outer_likelihoods[-1], msg_delta))
                return True
            else:
                return False
        else:
            return False

    def likelihood_changed(self):
        if len(self.likelihoods) > 1:
            diff = self.likelihoods[-1] - self.likelihoods[-2]
            diff_norm = abs(diff / self.likelihoods[-1])
            print(diff_norm)
            return diff_norm > 1e-2
        else: # no history so yes
            return False

    def likelihood_improved(self):
        """."""
        if len(self.em_likelihoods) > 1:
            if self.likelihoods[-1] < self.em_likelihoods[-1][2]:
                ll_iter_delta = self.abs_scale_factor_change(
                    self.likelihoods[-1], self.em_likelihoods[-1][2])
                if ll_iter_delta > self.ll_tol:
                    print("Loglikelikelihood has decreased ({}->{}, x{}), "
                          "keep iterating".format(
                        self.em_likelihoods[-1][2], self.likelihoods[-1],
                        ll_iter_delta))
                    return False
                else:
                    print("Loglikelikelihood has decreased ({}->{}), "
                          "but only by x{} (<x{}), "
                          "assume numerical precision issue and continue".format(
                        self.em_likelihoods[-1][2], self.likelihoods[-1],
                        ll_iter_delta, self.ll_tol))
                    return True
            else:
                return True
        else: # no history so yes
            return True

    @staticmethod
    def mvn_kl_divergence(mu1, sig1, mu2, sig2):
        try:
            mu_diff = mu2 - mu1
            dist = mu_diff.T.dot(np.linalg.solve(sig2, mu_diff))[0,0]
            log_det_ratio = np.log(np.linalg.det(sig2) / np.linalg.det(sig1))
            trace_ratio = np.trace(np.linalg.solve(sig2, sig1))
        except:
            print(mu1, sig1, mu2, sig2)
            raise
        return 0.5 * (log_det_ratio + trace_ratio + dist - mu1.shape[0])

    def _calc_trajectory_divergence(self):
        return sum([self.mvn_kl_divergence(c.mu_x3, c.sig_x3, c.mu_x3_m, c.sig_x3_m)
                    for c in self.cells])

    def _calc_posterior_trajectory_divergence(self):
        if self.cells[0].mu_x3_m_prev is not None:
            return sum([self.mvn_kl_divergence(c.mu_x3_m_prev, c.sig_x3_m_prev, c.mu_x3_m, c.sig_x3_m)
                        for c in self.cells])
        else:
            return 1000.0

    def calc_trajectory_diverence(self):
        self.msg_state_divergence.append(self._calc_trajectory_divergence())

    def posterior_trajectory_divergence(self):
        self.linearized_state_divergence.append(self._calc_posterior_trajectory_divergence())

    def state_divergence_converged(self, divergences, n, name):
        """."""
        if n > 0:
            msg_delta = self.abs_scale_factor_change(
                divergences[-1], divergences[-2])
            if msg_delta < self.msg_tol:
                print("{} converged after {}? delta {}->{} : {}".format(
                    name, n, divergences[-2], divergences[-1], msg_delta))
                return True
            else:
                print("{} delta {} < {}, {}->{}".format(
                    name, msg_delta, self.msg_tol,
                    divergences[-2], divergences[-1]))
                return False
        else:
            return False

    def _calc_trajectory_distance(self):
        return sum([abs(c.mu_x3 -c.mu_x3_m) for c in self.cells]) / self.H

    def update_prev_trajectory(self):
        for c in self.cells:
            c.mu_x3_m_prev = np.copy(c.mu_x3_m)
            c.sig_x3_m_prev = np.copy(c.sig_x3_m)

    def _forward_backward_msgs(self):
        """Note, there has been experiments with multiple msg iterations
        (like AICO), but so far the added computation hasn't been justified
        in results. Better to just have better priors."""
        self.policy_valid = False
        gap_bound = 2
        for n in range(self.msg_iter):
            self._forward_msgs()
            self._backward_msgs()
            if SIMULATE:
                self._simulate() # TODO really?
            self.calc_gap()
            break

    def costs_consistent(self):
        """We want to keep the distance between the prior and
        posterior trajectory as small as possible."""
        if len(self.costs_m) > 0:
            c_diff = self.costs_m[-1] - self.costs_p[-1]
            c_diff_norm = abs(c_diff / self.costs_m[-1])
            if len(self.cost_diff_norms) > 0:
                bounded = c_diff_norm <= self.cost_diff_norms[-1] * 1.001
            else:
                bounded = True
            self.cost_diff_norms.append(c_diff_norm)
            return bounded
        else:
            return True

    def learn_msgs(self):
        self.em_iter += 1
        self._forward_backward_msgs() # E Step
        self._maximize() # M Step
        self._update_priors() # M Step

    def plot_alphas(self, dir_name=None, filename=""):
        f = plt.figure()
        a_tv = np.asarray(self.alphas_tv).reshape(-1, self.H)
        # plt.plot(a_tv, '.')
        plt.plot(self.alphas, 'o-', label="actual")
        plt.plot(self.alphas_desired, 'o-', label="desired")
        plt.grid(True)
        plt.legend()
        plt.title("Sigma Xi scale factor over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("1 / Alpha")
        if dir_name:
            plt.savefig(os.path.join(dir_name, "alpha_{}.png".format(filename)),
                        bbox_inches='tight', format='png')
            plt.close(f)

    def plot_policy_entropy(self, dir_name=None, filename=""):
        f = plt.figure()
        _iter = [p[0] for p in self.policy_entropy]
        _ent = [p[1] for p in self.policy_entropy]
        plt.plot(_iter, _ent, 'o-')
        plt.grid(True)
        plt.title("Policy Entropy")
        plt.xlabel("Iterations")
        plt.ylabel("Total Policy Entropy")
        if dir_name:
            plt.savefig(os.path.join(dir_name, "entropy_{}.png".format(filename)),
                        bbox_inches='tight', format='png')
            plt.close(f)

    def plot_traj_divergence(self, dir_name, filename):
        f, a = plt.subplots(2,1)
        pts = [i-1 for i in self.msg_state_divergence_converge]
        div_pts = [self.msg_state_divergence[i] for i in pts]
        a[0].plot(self.msg_state_divergence, 'bo-')
        a[0].plot(pts, div_pts, 'ko')
        a[0].grid()
        a[1].plot(self.linearized_state_divergence, 'o-')
        a[1].grid()
        a[0].set_ylabel("Prior Posterior State Divergence")
        a[1].set_ylabel("Posterior State Divergence over Linearization")
        a[0].set_xlabel("Msg Iterations")
        a[1].set_xlabel("Linearization Iterations")
        if dir_name:
            plt.savefig(os.path.join(dir_name, "divergence_{}.png".format(filename)),
                        bbox_inches='tight', format='png')
            plt.close(f)

    def plot_gap(self, dir_name, filename):
        f = plt.figure()
        plt.plot(self.traj_gaps, 'o-')
        plt.grid(True)
        plt.title("L2 distance between filtered and marginal state trajectory")
        plt.xlabel("Iterations")
        plt.ylabel("L2 Norm")
        if dir_name:
            plt.savefig(os.path.join(dir_name, "gap_{}.png".format(filename)),
                        bbox_inches='tight', format='png')
            plt.close(f)

    def plot_msg_iters(self, dir_name, filename):
        f = plt.figure()
        plt.plot(self.msg_iters, 'o-')
        plt.grid(True)
        plt.title("E-Step iterations")
        plt.xlabel("EM Iterations")
        plt.ylabel("E Iteration")
        if dir_name:
            plt.savefig(os.path.join(dir_name, "msg_iter_{}.png".format(filename)),
                        bbox_inches='tight', format='png')
            plt.close(f)

    def plot_cost(self, dir_name, filename):
        f, axes = plt.subplots(4,1)
        a = axes[0]
        _a = a.twinx()
        a.set_title("Quadratic Cost and LogLikelihood over Message Iterations")
        a.plot(self.costs_p, 'ro-', label="Predictive")
        a.plot(self.costs_m, 'rx-', label="Marginal")
        a.plot(self.costs_m, 'rs-', label="Simulate")

        a.set_ylabel("Cost", color='r')
        a.legend(loc="lower left")
        a.grid()
        iter_ll_pos = [ll[0]-1 for ll in self.em_likelihoods]
        iter_ll = [ll[2] for ll in self.em_likelihoods]
        _a.plot(self.likelihoods, 'b.-', label="Message Iter Full")
        _a.plot(iter_ll_pos, iter_ll, 'ko', label="EM Iter")
        _a.grid()
        _a.legend(loc="lower right")
        _a.set_ylabel("Log Likelikelihood", color='b')

        _a = axes[1].twinx()
        _a.plot(self.likelihoods_xu, 'co-', label="Message Iter State Action")
        _a.grid()
        _a.legend(loc="lower right")
        _a.set_ylabel("Log Likelikelihood", color='b')

        _a = axes[2].twinx()
        _a.plot(self.likelihoods_z, 'cx-', label="Message Iter Observation")
        _a.grid()
        _a.legend(loc="lower right")
        _a.set_ylabel("Log Likelikelihood", color='b')

        axes[3].plot(self.cost_diff_norms, '.-')
        axes[3].set_ylabel("Normalised Cost Difference")
        axes[-1].set_xlabel("Message Iterations")
        if dir_name:
            plt.savefig(os.path.join(dir_name, "msg_cost_{}.png".format(filename)),
                        bbox_inches='tight', format='png')
            if PLOT_PDF:
                plt.savefig(os.path.join(dir_name, "msg_cost_{}.pdf".format(filename)),
                            bbox_inches='tight', format='pdf')
            plt.close(f)

    def plot_cost_all(self, dir_name, filename):
        f, axes = plt.subplots(3,1)
        a = axes[0]
        _a = a.twinx()
        a.set_title("Quadratic Cost and LogLikelihood over Iterations")
        a.plot(self.costs_p_all, 'ro-', label="Predictive")
        a.plot(self.costs_m_all, 'rx-', label="Marginal")

        a.set_ylabel("Cost", color='r')
        a.legend(loc="lower left")
        a.grid()
        iter_ll_pos = [ll[0]-1 for ll in self.em_likelihoods]
        iter_ll = [ll[2] for ll in self.em_likelihoods]
        _a.plot(self.likelihoods_all, 'b.-', label="Message Iter Full")
        _a.grid()
        _a.legend(loc="lower right")
        _a.set_ylabel("Log Likelikelihood", color='b')

        _a = axes[1].twinx()
        _a.plot(self.likelihoods_xu_all, 'co-', label="Message Iter State Action")
        _a.grid()
        _a.legend(loc="lower right")
        _a.set_ylabel("Log Likelikelihood", color='b')

        axes[-1].set_xlabel("EM Iterations")
        if dir_name:
            plt.savefig(os.path.join(dir_name, "msg_cost_all_{}.png".format(filename)),
                        bbox_inches='tight', format='png')
            if PLOT_PDF:
                plt.savefig(os.path.join(dir_name, "msg_cost_all_{}.pdf".format(filename)),
                            bbox_inches='tight', format='pdf')
            plt.close(f)

    def plot_em_cost(self, dir_name, filename):
        f, a = plt.subplots(1,1)
        em_pos = [ll[0]-1 for ll in self.em_likelihoods]
        em_iter = [ll[1]-1 for ll in self.em_likelihoods]
        em_ll = [ll[2] for ll in self.em_likelihoods]
        em_cost_p = [self.costs_p[i] for i in em_pos]
        em_cost_m = [self.costs_m[i] for i in em_pos]
        if SIMULATE:
            em_cost_s = [self.costs_s[i] for i in em_pos]
        _a = a.twinx()
        a.set_title("Quadratic Cost and LogLikelihood over EM Iterations")
        a.plot(em_iter, em_cost_p, 'ro-', label="Predictive")
        a.plot(em_iter, em_cost_m, 'rx-', label="Marginal")
        if SIMULATE:
            a.plot(em_iter, em_cost_s, 'rs-', label="Simulate")
        a.set_xlabel("EM Iterations")
        a.set_ylabel("Cost", color='r')
        a.legend()
        _a.plot(em_iter, em_ll, 'b.-')
        _a.set_ylabel("Log Likelikelihood", color='b')
        if dir_name:
            plt.savefig(os.path.join(dir_name, "em_cost_{}.png".format(filename)),
                        bbox_inches='tight', format='png')
            if PLOT_PDF:
                plt.savefig(os.path.join(dir_name, "em_cost_{}.pdf".format(filename)),
                            bbox_inches='tight', format='pdf')
            if PLOT_TIKZ:
                matplotlib2tikz.save(
                    os.path.join(dir_name, "em_cost_{}.tex".format(filename)))
            plt.close(f)

    def plot_observed_traj(self, filename="", dir_name=None):
        y_m = np.asarray([c.mu_z0_m for c in self.cells]).squeeze()
        f, _a = plt.subplots(self.sys.dim_y, 1)
        for i in range(self.sys.dim_y):
            g = self.sys.sg[i] * np.ones((self.H,))
            _a[i].set_ylabel(self.sys.y_key[i])
            _a[i].plot(y_m[:, i])
            _a[i].plot(g, 'k--')
        _a[0].set_title("Observation")
        if dir_name is not None:
            plt.savefig(os.path.join(dir_name,
                "observation_{}.png".format(filename)),
                bbox_inches='tight', format='png')
            plt.close(f)

    def plot_system_dynamics(self, filename="", dir_name=None):
        Aeig_sys = np.asarray([np.sort(np.linalg.eig(c.A)[0]) for c in self.cells]).squeeze()
        Aeig_sys_m = np.absolute(Aeig_sys)
        Aeig_sys_p = np.angle(Aeig_sys)
        Bs = np.asarray([c.B for c in self.cells])
        _a = np.asarray([c.a for c in self.cells])
        E = np.asarray([c.E for c in self.cells])
        f, a = plt.subplots(5 ,1)
        a[0].set_title("Linearized Dynamics")
        a[0].plot(Aeig_sys_m)
        a[0].set_ylabel("A Eigenvalue Magnitude")
        a[1].plot(Aeig_sys_p)
        a[1].set_ylabel("A Eigenvalue Phase")
        a[2].set_ylabel("B Elements")
        for i in range(self.sys.dim_x):
            for j in range(self.sys.dim_u):
                a[2].plot(Bs[:, i, j], label="B{}{}".format(i, j))
        a[2].legend()
        for i in range(self.sys.dim_x):
            a[3].plot(_a[:, i], label="a{}".format(i))
        a[3].set_ylabel("a Elements")
        a[3].legend()
        for j in range(self.sys.dim_x):
            for i in range(self.sys.dim_y):
                a[4].plot(E[:, i, j], label="E{}{}".format(i, j))
        a[4].set_ylabel("E Elements")
        a[4].legend(ncol=self.sys.dim_x)
        a[-1].set_xlabel("Timesteps")
        if dir_name is not None:
            plt.savefig(os.path.join(dir_name,
                "system_dynamics_{}.png".format(filename)),
                bbox_inches='tight', format='png')
            plt.close(f)

    def plot_controller(self, filename="", dir_name=None,  lqr_compare=False):
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
            a[i].plot(t, um_m[:, i], 'k.-', label="marginal")
            a[i].plot(t, u_pol_K[:, i], 'g.-', label="Feedback")
            a[i].plot(t, u_pol_k[:, i], 'b.-', label="Feedforward")
            a[i].plot(t, u_pol[:, i], 'r.', label="Controller")
            if lqr_compare:
                a[i].plot(t, u_lqr[:, i], 'y.', label="LQR")
            a[i].legend()
            idx += 1

        for i in range(self.sys.dim_u):
            for j in range(self.sys.dim_x):
                a[idx].plot(t, K[:, i, j], label="K{}{}".format(i, j))
                if lqr_compare:
                    a[idx].plot(t, K_lqr[:, i, j], label="K{}{}_lqr".format(i, j))
        a[idx].legend()
        idx += 1
        for i in range(self.sys.dim_u):
            a[idx].plot(t, k[:, i], label="k{}".format(i))
            if lqr_compare:
                a[idx].plot(t, k_lqr[:, i], label="k{}_lqr".format(i))
        a[idx].legend()

        if dir_name is not None:
            plt.savefig(os.path.join(dir_name,
                "controller_{}.png".format(filename)),
                bbox_inches='tight', format='png')
            plt.close(f)

    def plot_ricatti(self, filename="", dir_name=None):
        lam_x3 = np.asarray([c.lambda_x3_b for c in self.cells]).squeeze() * self.alpha
        nu_x3 = np.asarray([c.nu_x3_b for c in self.cells]).squeeze() * self.alpha
        f,a = plt.subplots(2,1)
        a[0].set_title("lam x3 b")
        for i in range(self.sys.dim_x):
            for j in range(self.sys.dim_x):
                a[0].plot(lam_x3[:, i, j], label="lambda x b {}{}".format(i, j))
        a[0].legend()
        a[1].set_title("nu x3 b")
        for i in range(self.sys.dim_x):
            a[1].plot(nu_x3[:, i], label="nu x b {}".format(i))
        a[1].legend()
        if dir_name is not None:
            plt.savefig(os.path.join(dir_name,
                "ricatti_{}.png".format(filename)),
                bbox_inches='tight', format='png')
            plt.close(f)

    def plot_uncertainty(self, filename="", dir_name=None):
        t = range(self.H)
        sigEta = np.asarray([c.sigEta for c in self.cells]).squeeze()
        sigu1 = np.asarray([c.sig_u1_f for c in self.cells])
        sigx1 = np.asarray([c.sig_x1_f for c in self.cells]).squeeze()
        sigx2 = np.asarray([c.sig_x2_f for c in self.cells]).squeeze()
        sigu2 = np.asarray([c.sig_u2_f for c in self.cells]).squeeze()
        sigx0_m = np.asarray([c.sig_x0_m for c in self.cells]).squeeze()
        sigu0_m = np.asarray([c.sig_u0_m for c in self.cells])

        f, a = plt.subplots(self.sys.dim_x + self.sys.dim_u, 1)
        plt.title("Uncertainties")
        for i in range(self.sys.dim_x):
            a[i].plot(t, sigx1[:, i, i], 'b.-', label="sig x1 fwd")
            a[i].plot(t, sigx2[:, i, i], 'y.-', label="sig x2 fwd")
            a[i].plot(t, sigu2[:, i, i], 'c.-', label="sig u2 f")
            a[i].plot(t, sigx0_m[:, i, i], 'g.-', label="sig x0 mrg")
            a[i].legend(loc='lower left')
            a[i].set_ylabel('SigX1', color='b')
            _a = a[i].twinx()
            _a.plot(t, sigEta[:, i, i], 'r.-', label="sigEta")
            _a.set_ylabel('SigV', color='r')
            _a.legend(loc='lower right')
        for i in range(self.sys.dim_u):
            j = self.sys.dim_x + i
            a[j].plot(t, sigu0_m[:, i, i], 'g.-', label="sig u0 mrg")
            a[j].plot(t, sigu1[:, i, i], 'b.-', label="sig u1 fwd")
            a[j].set_ylabel('SigU')
            a[j].legend()
        if dir_name is not None:
            plt.savefig(os.path.join(dir_name,
                "uncertainties_{}.png".format(filename)),
                bbox_inches='tight', format='png')
            plt.close(f)

    def get_local_linear_policy(self):
        if not self.policy_valid:
            self._forward_msgs()
            self._backward_msgs()
            self._backward_ricatti_msgs()
        K = np.asarray([c.K for c in self.cells]).reshape((-1, self.sys.dim_u, self.sys.dim_x))
        k = np.asarray([c.k for c in self.cells]).reshape((-1, self.sys.dim_u))
        sigk = np.asarray([c.sigK for c in self.cells]).reshape((-1, self.sys.dim_u, self.sys.dim_u))
        return K, k, sigk

    def get_marginal_input(self):
        return np.asarray([c.mu_u0_m for c in self.cells])

    def get_marginal_trajectory(self):
        x = np.asarray([c.mu_x0_m for c in self.cells])
        u = np.asarray([c.mu_u0_m for c in self.cells])
        s = np.hstack((x, u)).reshape((-1, self.sys.dim_xt))
        return s

    def get_marginal_observed_trajectory(self):
        z = [c.mu_z0_m for c in self.cells]
        zT = self.sys.observe(
            self.cells[-1].mu_x3_m,
             # there is technically no u here, so go
             # with zero
            np.zeros((self.sys.dim_u, 1)))[0]
        z.append(zT)
        return np.asarray(z).reshape((-1, self.sys.dim_y))

    def reset_priors(self):
        self.alpha = self.alpha_base
        for c in self.cells:
            c.mu_u0 = np.copy(c.mu_u0_base)
            c.sig_u0 = np.copy(c.sig_u0_base)
            c.sigXi = self.sigXi
            # signal for linearizations to be recalculated in during prior
            c.linearized = False

        self.reset_metrics()

    def reset_metrics(self, extend=True):
        if extend:
            self.costs_m_all.extend(self.costs_m)
            self.costs_p_all.extend(self.costs_p)
            self.likelihoods_all.extend(self.likelihoods)
            self.likelihoods_xu_all.extend(self.likelihoods_xu)

        self.likelihoods = []
        self.likelihoods_xu = []
        self.likelihoods_z = []
        self.em_likelihoods = []
        self.outer_likelihoods = []
        self.linearized_state_divergence = []
        self.msg_state_divergence = []
        self.msg_state_divergence_converge = []
        self.em_cost = []
        self.costs_p = []
        self.costs_m = []
        self.costs_s = []
        self.traj_gaps = []
        self.policy_entropy = []

    def save_traj(self, res_dir):
        x = np.asarray([c.mu_x0_m for c in self.cells])
        u = np.asarray([c.mu_u0_m for c in self.cells])
        z = np.asarray([c.mu_z0_m for c in self.cells])
        np.save(os.path.join(res_dir, "x_marg.npy"), x)
        np.save(os.path.join(res_dir, "u_marg.npy"), u)
        np.save(os.path.join(res_dir, "z_marg.npy"), z)

    def converged(self):
        delta_tol_pcnt = 0.005
        if len(self.costs_m) > 2:
            delta_pcnt = abs(self.costs_m[-1] - self.costs_m[-2]) / self.costs_m[-1]
            return delta_pcnt < delta_tol_pcnt
        else:
            return False

    def save(self, path, name):
        filename = "i2c_{}.pkl".format(name)
        with open(os.path.join(path, filename), 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            obj = dill.load(f)
        return obj