import numpy as np
import scipy as sc
from scipy import optimize
import matplotlib.pyplot as plt

from trajopt.gps.mbgps import MBGPS
from trajopt.gps.objects import (Gaussian, QuadraticStateValue,
                                 QuadraticStateActionValue,
                                 AnalyticalLinearGaussianDynamics,
                                 LinearGaussianControl,
                                 AnalyticalQuadraticCost)

class GuidedPolicySearch(MBGPS):

    def __init__(self, model, reward, horizon, kl_bound,
                 u_lim, init_ctl_sigma, init_noise, activation='last'):

        self.env = model

        self.env_dyn = self.env.dynamics
        self.env_noise = lambda x, u: self.env.sigV
        self.env_cost = reward
        self.env_init = self.env.init

        self.ulim = u_lim

        self.nb_xdim = self.env.dim_x
        self.nb_udim = self.env.dim_u
        self.nb_steps = horizon

        # total kl over traj.
        # total kl over traj.
        self.kl_base = kl_bound
        self.kl_bound = kl_bound

        # kl mult.
        self.kl_mult = 1.
        self.kl_mult_min = 0.1
        self.kl_mult_max = 5.0
        self.alpha = np.array([-100.])

        # create state distribution and initialize first time step
        self.xdist = Gaussian(self.nb_xdim, self.nb_steps + 1)
        self.xdist.mu[..., 0], self.xdist.sigma[..., 0] = self.env_init()

        self.udist = Gaussian(self.nb_udim, self.nb_steps)
        self.xudist = Gaussian(self.nb_xdim + self.nb_udim, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        self.dyn = AnalyticalLinearGaussianDynamics(self.env_init, self.env_dyn, self.env_noise, self.nb_xdim, self.nb_udim, self.nb_steps)
        self.ctl = LinearGaussianControl(self.nb_xdim, self.nb_udim, self.nb_steps, init_ctl_sigma)
        self.ctl.kff = init_noise * np.random.randn(self.nb_udim, self.nb_steps)

        if activation == 'all':
            self.activation = np.ones((self.nb_steps + 1,), dtype=np.int64)
        else:
            self.activation = np.zeros((self.nb_steps + 1, ), dtype=np.int64)
            self.activation[-1] = 1
        self.cost = AnalyticalQuadraticCost(self.env_cost, self.nb_xdim, self.nb_udim, self.nb_steps + 1)

    def plot_trajectory(self):
        f,a = plt.subplots(self.nb_xdim + self.nb_udim, 1)
        t = range(self.nb_steps)
        idx = 0
        for i in range(self.nb_xdim):
            dx = 3. * self.xdist.sigma[i, i, :-1]
            xl = self.xdist.mu[i, :-1] - dx
            xu =  self.xdist.mu[i, :-1] + dx
            a[idx].plot(self.xdist.mu[i, :-1], 'b.-')
            a[idx].fill_between(t, xl, xu, where=xu>=xl,
                              facecolor='b', alpha=0.4)
            idx += 1
        for i in range(self.nb_udim):
            du = 3. * self.udist.sigma[i, i, :]
            ul = self.udist.mu[i, :] - du
            uu =  self.udist.mu[i, :] + du
            a[idx].plot(self.udist.mu[i, :], 'r.-')
            a[idx].fill_between(t, ul, uu, where=uu>=ul,
                              facecolor='r', alpha=0.4)
            idx += 1
