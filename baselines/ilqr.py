import numpy as np
import scipy as sc
from scipy import optimize
import matplotlib.pyplot as plt

from trajopt.ilqr.ilqr import iLQR
from trajopt.ilqr.objects import (QuadraticStateValue,
                                 QuadraticStateActionValue,
                                 AnalyticalLinearDynamics,
                                 LinearControl,
                                 AnalyticalQuadraticCost)

class IterativeLqr(iLQR):

    def __init__(self, env, cost, horizon, u_lim, init_noise = 1e-2,
                 alphas=np.power(10., np.linspace(0, -3, 11)),
                 lmbda=1., dlmbda=1.,
                 min_lmbda=1.e-6, max_lmbda=1.e3, mult_lmbda=1.6,
                 tolfun=1.e-7, tolgrad=1.e-4, min_imp=0., reg=1,
                 activation='all'):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.dynamics
        self.env_cost = cost
        # self.env_init = lambda: self.env.init()[0] # sorry
        self.env_init = self.env.init

        self.ulim = u_lim

        self.nb_xdim = self.env.dim_x
        self.nb_udim = self.env.dim_u
        self.nb_steps = horizon

        # backtracking
        self.alphas = alphas
        self.lmbda = lmbda
        self.dlmbda = dlmbda
        self.min_lmbda = min_lmbda
        self.max_lmbda = max_lmbda
        self.mult_lmbda = mult_lmbda

        # regularization type
        self.reg = reg

        # minimum relative improvement
        self.min_imp = min_imp

        # stopping criterion
        self.tolfun = tolfun
        self.tolgrad = tolgrad

        # reference trajectory
        self.xref = np.zeros((self.nb_xdim, self.nb_steps + 1))
        self.xref[..., 0] = self.env_init()[0]

        self.uref = np.zeros((self.nb_udim, self.nb_steps))

        self.vfunc = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        self.dyn = AnalyticalLinearDynamics(self.env_init, self.env_dyn, self.nb_xdim, self.nb_udim, self.nb_steps)
        self.ctl = LinearControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        self.ctl.kff = init_noise * np.random.randn(self.nb_udim, self.nb_steps)

        # activation of cost function
        if activation == 'all':
            self.activation = np.ones((self.nb_steps + 1,), dtype=np.int64)
        else:
            self.activation = np.zeros((self.nb_steps + 1, ), dtype=np.int64)
            self.activation[-1] = 1

        self.cost = AnalyticalQuadraticCost(self.env_cost, self.nb_xdim, self.nb_udim, self.nb_steps + 1)

        self.last_objective = - np.inf

    def plot_trajectory(self):
        f,a = plt.subplots(self.nb_xdim + self.nb_udim, 1)
        idx = 0
        for i in range(self.nb_xdim):
            a[idx].plot(self.xref[i, :], 'b.-')
            idx += 1
        for i in range(self.nb_udim):
            a[idx].plot(self.uref[i, :], 'r.-')
            idx += 1
