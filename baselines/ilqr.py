import os
import numpy as np
import matplotlib.pyplot as plt

from trajopt.ilqr.ilqr import iLQR
from trajopt.ilqr.objects import (
    QuadraticStateValue,
    QuadraticStateActionValue,
    AnalyticalLinearDynamics,
    LinearControl,
    AnalyticalQuadraticCost,
)


class IterativeLqr(iLQR):
    def __init__(
        self,
        env,
        cost,
        horizon,
        u_lim,
        init_noise=1e-2,
        alphas=np.power(10.0, np.linspace(0, -3, 11)),
        lmbda=1.0,
        dlmbda=1.0,
        min_lmbda=1.0e-6,
        max_lmbda=1.0e3,
        mult_lmbda=1.6,
        tolfun=1.0e-7,
        tolgrad=1.0e-4,
        min_imp=0.0,
        reg=1,
        activation="all",
        dir_name=None,
    ):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.predict_1d
        self.env_cost = cost
        self.env_init = self.env.init()[0]

        self.ulim = u_lim

        self.dm_state = self.env.dim_x
        self.dm_act = self.env.dim_u
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
        self.xref = np.zeros((self.dm_state, self.nb_steps + 1))
        self.xref[..., 0] = self.env_init

        self.uref = np.zeros((self.dm_act, self.nb_steps))

        self.vfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(
            self.dm_state, self.dm_act, self.nb_steps
        )

        self.dyn = AnalyticalLinearDynamics(
            self.env_dyn, self.dm_state, self.dm_act, self.nb_steps
        )
        self.ctl = LinearControl(self.dm_state, self.dm_act, self.nb_steps)
        self.ctl.kff = init_noise * np.random.randn(self.dm_act, self.nb_steps)

        # activation of cost function
        self.weighting = np.ones((self.nb_steps + 1,))

        self.cost = AnalyticalQuadraticCost(
            self.env_cost, self.dm_state, self.dm_act, self.nb_steps + 1
        )

        self.last_objective = -np.inf

        self.feedforward = False

        self.dir_name = dir_name

    def forward_pass(self, ctl, alpha):
        state = np.zeros((self.dm_state, self.nb_steps + 1))
        action = np.zeros((self.dm_act, self.nb_steps))
        cost = np.zeros((self.nb_steps + 1,))

        state[..., 0] = self.env_init
        for t in range(self.nb_steps):
            if self.feedforward:
                _act = ctl.action(state, alpha, state, self.uref, t)
            else:
                _act = ctl.action(state, alpha, self.xref, self.uref, t)

            action[..., t] = np.clip(_act, self.ulim[0], self.ulim[1])
            cost[..., t] = self.env_cost(
                state[..., t], action[..., t], self.weighting[t]
            )
            state[..., t + 1] = self.env_dyn(state[..., t], action[..., t])

        cost[..., -1] = self.env_cost(
            state[..., -1], np.zeros((self.dm_act,)), self.weighting[-1]
        )
        return state, action, cost

    def plot_trajectory(self, filename=""):
        f, a = plt.subplots(self.dm_state + self.dm_act, 1)
        idx = 0
        for i in range(self.dm_state):
            name = self.env.key[idx]
            unit = self.env.unit[idx]
            a[idx].plot(self.xref[i, :], "b.-")
            a[idx].set_ylabel("{} ({})".format(name, unit))
            idx += 1
        for i in range(self.dm_act):
            name = self.env.key[idx]
            unit = self.env.unit[idx]
            a[idx].plot(self.uref[i, :], "r.-")
            a[idx].set_ylabel("{} ({})".format(name, unit))
            idx += 1
        a[-1].set_xlabel("Timesteps")
        if self.dir_name is not None:
            plt.savefig(
                os.path.join(self.dir_name, "ilqr_traj_{}.png".format(filename)),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)
