"""
Experiment to show i2c can do LQR!
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

DIR_NAME = os.path.dirname(__file__)


from i2c.i2c import I2cGraph, PLOT_TIKZ
from i2c.env import make_env
from i2c.model import make_env_model
from i2c.policy.linear import TimeIndexedLinearGaussianPolicy
from i2c.utils import (
    make_results_folder,
    finite_horizon_lqr,
)

import scripts.experiments.linear_known as experiment


def plot_trajectory(i2c, x_lqr, u_lqr, dir_name=None):
    f, a = plt.subplots(3, 1)

    t = range(i2c.H)
    a[0].set_title("State Trajectory")
    a[0].set_ylabel("$x_1$")
    a[1].set_ylabel("$x_2$")
    a[2].set_ylabel("$u$")
    a[2].set_xlabel("$t$")

    xu_i2c = i2c.get_marginal_state_action()
    xu_i2c_f = i2c.get_state_action_prior()
    a[0].plot(t, x_lqr[:, 0], "k+-", label="LQR")
    a[0].plot(t, xu_i2c[:, 0], "c--", label="I2C posterior")
    a[0].plot(t, xu_i2c_f[:, 0], "m--", label="I2C prior")
    a[1].plot(t, x_lqr[:, 1], "k+-")
    a[1].plot(t, xu_i2c[:, 1], "c--")
    a[1].plot(t, xu_i2c_f[:, 1], "m--")
    a[2].plot(t, u_lqr[:, 0], "k+-")
    a[2].plot(t, xu_i2c[:, 2], "c--")
    a[2].plot(t, xu_i2c_f[:, 2], "m--")

    a[0].legend()

    if dir_name is not None:
        plt.savefig(
            os.path.join(dir_name, "trajectory.png"), bbox_inches="tight", format="png"
        )
        tikzplotlib.save(os.path.join(dir_name, "trajectory.tex"))
        plt.close(f)


def plot_controller(i2c, u_lqr, K_lqr, k_lqr, dir_name=None):
    t = range(i2c.H)
    K, k, _ = i2c.get_local_linear_policy()

    f, a = plt.subplots(2, 1)

    a[0].set_title("Time-varying Linear Controller")
    a[0].set_ylabel("Feedback Gains, $\mathbold{{K}}$")
    a[1].set_ylabel("Feedforward Gains, $\mathbold{{k}}$")
    _K = K.reshape((i2c.H, -1))
    _K_lqr = K_lqr.reshape((i2c.H, -1))
    for i in range(_K.shape[1]):
        a[0].plot(t, _K_lqr[:, i], "k+-", label="LQR" if i == 0 else "_")
        a[0].plot(t, _K[:, i], "rx", label="I2C" if i == 0 else "_")
    a[0].legend()
    for i in range(i2c.sys.dim_u):
        a[1].plot(t, k_lqr[:, i], "k+-", label="LQR" if i == 0 else "_")
        a[1].plot(t, k[:, i], "rx", label="I2C" if i == 0 else "_")
    a[1].legend()
    a[1].set_xlabel("Timesteps")

    if dir_name is not None:
        plt.savefig(
            os.path.join(dir_name, "controller.png"), bbox_inches="tight", format="png"
        )
        tikzplotlib.save(os.path.join(dir_name, "controller.tex"))
        plt.close(f)


def plot_value_function(i2c, P, p, dir_name=None):
    f, a = plt.subplots(2, 1)
    _P = P.reshape((-1, i2c.sys.dim_x * 2))
    lam_x3 = np.asarray([c.lambda_x3_b for c in i2c.cells]) * i2c.alpha
    nu_x3 = np.asarray([c.nu_x3_b for c in i2c.cells]) * i2c.alpha
    _lam_x3 = lam_x3.reshape((-1, i2c.sys.dim_x * 2))
    for i in range(i2c.sys.dim_x * 2):
        a[0].plot(_P[:, i], "k+-", label=r"$\mathbold{{P}}$" if i == 0 else "_")
        a[0].plot(
            _lam_x3[:, i],
            "rx",
            label=r"$\mathbold{{\Lambda}}_{{\overleftarrow{{x}}}}$" if i == 0 else "_",
        )

    a[0].legend()
    a[0].set_title("Value Function parameters")
    a[0].set_ylabel("Quadratic Weights")
    a[1].set_ylabel("Linear Weights")
    a[1].set_xlabel("Timesteps")
    for i in range(i2c.sys.dim_x):
        a[1].plot(p[:, i], "k+-", label=r"$\mathbold{{p}}$" if i == 0 else "_")
        a[1].plot(
            -nu_x3[:, i],
            "rx",
            label=r"$-\mathbold{{\nu}}_{{\overleftarrow{{x}}}}$" if i == 0 else "_",
        )
    a[1].legend()
    if dir_name is not None:
        plt.savefig(
            os.path.join(dir_name, "value.png"), bbox_inches="tight", format="png"
        )
        tikzplotlib.save(os.path.join(dir_name, "value.tex"))
        plt.close(f)


def main():
    res_dir = make_results_folder("i2c_lqr_equivalence", 0, "", release=True)

    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT, experiment.MODEL)

    experiment.N_INFERENCE = 1

    # redefine linear system
    model.xag = 10 * np.ones((env.dim_x, 1))
    model.zg_term = 10 * np.ones((env.dim_x, 1))
    model.a = model.xag - model.A @ model.xag
    env.a = model.a
    env.sig_eta = 0.0 * np.eye(env.dim_x)
    ug = np.zeros((env.dim_u,))

    x_lqr, u_lqr, K_lqr, k_lqr, cost_lqr, P, p = finite_horizon_lqr(
        experiment.N_DURATION,
        model.A,
        model.a[:, 0],
        model.B,
        experiment.INFERENCE.Q,
        experiment.INFERENCE.R,
        model.x0[:, 0],
        model.xag[:, 0],
        ug,
        model.dim_x,
        model.dim_u,
    )
    from i2c.exp_types import CubatureQuadrature

    i2c = I2cGraph(
        sys=model,
        horizon=experiment.N_DURATION,
        Q=experiment.INFERENCE.Q,
        R=experiment.INFERENCE.R,
        Qf=experiment.INFERENCE.Qf,
        alpha=1e-5,  # 1e-6,
        alpha_update_tol=experiment.INFERENCE.alpha_update_tol,
        mu_u=np.zeros((experiment.N_DURATION, 1)),
        sig_u=1e2 * np.eye(1),
        mu_x_terminal=None,
        sig_x_terminal=None,
        inference=experiment.INFERENCE.inference,
        res_dir=None,
    )
    i2c.use_expert_controller = False
    for c in i2c.cells:
        c.state_action_independence = True

    # EM iteration
    i2c._forward_backward_msgs()
    i2c.plot_traj(0, dir_name=res_dir, filename="lqr")

    # compute riccati terms
    i2c._backward_ricatti_msgs()

    plot_trajectory(i2c, x_lqr, u_lqr, dir_name=res_dir)
    plot_controller(i2c, u_lqr, K_lqr, k_lqr, dir_name=res_dir)
    plot_value_function(i2c, P, p, dir_name=res_dir)


if __name__ == "__main__":
    import matplotlib

    PLOT_TIKZ = True
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["figure.figsize"] = [10, 10]
    matplotlib.rcParams["legend.fontsize"] = 16
    matplotlib.rcParams["axes.titlesize"] = 22
    matplotlib.rcParams["axes.labelsize"] = 22
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
    matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amssymb}"]
    matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{fixmath}"]
    main()
    plt.show()
