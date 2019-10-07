"""
Experiment to show i2c can do LQR!
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib2tikz

DIR_NAME = os.path.dirname(__file__)

try:
    import pi2c
except ImportError:
    print("pi2c not installed, using local version")
    top_path = os.path.join(DIR_NAME, '..')
    sys.path.append(os.path.abspath(top_path))

from pi2c.i2c import I2cGraph, PLOT_TIKZ
from pi2c.env import make_env
from pi2c.model import make_env_model
from pi2c.policy.linear import TimeIndexedLinearGaussianPolicy
from pi2c.utils import (converged_list, TrajectoryData, profile, make_results_folder,
    configure_plots, TrajectoryEvaluator, finite_horizon_lqr)

import scripts.experiments.linear_known as experiment

def plot_controller(i2c, dir_name=None):
        t = range(i2c.H)
        K, k, _ = i2c.get_local_linear_policy()

        _, u_lqr, K_lqr, k_lqr = i2c.finite_horizon_lqr()

        f, a = plt.subplots(2, 1)

        a[0].set_title("Time-varying Linear Controller")
        a[0].set_ylabel("Feedback Gains, $\mathbold{{K}}$")
        a[1].set_ylabel("Feedforward Gains, $\mathbold{{k}}$")
        _K = K.reshape((i2c.H, -1))
        _K_lqr = K_lqr.reshape((i2c.H, -1))
        for i in range(_K.shape[1]):
            a[0].plot(t, _K_lqr[:, i],  'k+-',
                label="LQR" if i == 0 else "_")
            a[0].plot(t, _K[:, i],  'rx',
                label="I2C" if i == 0 else "_")
        a[0].legend()
        for i in range(i2c.sys.dim_u):
            a[1].plot(t, k_lqr[:, i],  'k+-',
                label="LQR" if i == 0 else "_")
            a[1].plot(t, k[:, i],  'rx',
                label="I2C" if i == 0 else "_")
        a[1].legend()
        a[1].set_xlabel("Timesteps")

        if dir_name is not None:
            plt.savefig(os.path.join(dir_name,
                "controller.png"),
                bbox_inches='tight', format='png')
            matplotlib2tikz.save(
                    os.path.join(dir_name, "controller.tex"))
            plt.close(f)

def main():


    res_dir = "_linear"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT,
                           experiment.MODEL)

    experiment.N_INFERENCE = 1 # 100
    model.xg = 10 * np.ones((env.dim_x,1))
    model.xag = model.xg
    model.a = model.xg - model.A.dot(model.xg)
    env.a = model.a.squeeze()  # aaaaah

    policy_lqr = TimeIndexedLinearGaussianPolicy(experiment.POLICY_COVAR,
        experiment.N_DURATION, model.dim_u, model.dim_x)
    policy_i2c = TimeIndexedLinearGaussianPolicy(experiment.POLICY_COVAR,
        experiment.N_DURATION, model.dim_u, model.dim_x)

    ug = np.zeros((env.dim_u,))
    x_lqr, u_lqr, K_lqr, k_lqr, cost_lqr, P, p = finite_horizon_lqr(
        experiment.N_DURATION, model.A, model.a.squeeze(), model.B, experiment.INFERENCE.Q, experiment.INFERENCE.R,
        model.x0, model.xg.squeeze(), ug, model.dim_x, model.dim_u)

    policy_lqr.K, policy_lqr.k =  K_lqr, k_lqr
    x, y, z = env.run(policy_lqr)
    env.plot_sim(x, None, "LQR actual")

    i2c = I2cGraph(
            model,
            experiment.N_DURATION,
            experiment.INFERENCE.Q,
            experiment.INFERENCE.R,
            1e-5,  # big alpha so control cost dominates
            experiment.INFERENCE.alpha_update_tol,
            experiment.INFERENCE.SIG_U,
            experiment.INFERENCE.msg_iter,
            experiment.INFERENCE.msg_tol,
            experiment.INFERENCE.em_tol,
            experiment.INFERENCE.backwards_contraction,
            None)

    traj_eval = TrajectoryEvaluator(experiment.N_DURATION, i2c.QR, i2c.sg)

    # getting policy runs messages
    policy_i2c.K, policy_i2c.k, _ = i2c.get_local_linear_policy()

    i2c.plot_traj("State-action Trajectories", lqr_compare=True,
        dir_name=res_dir)

    plot_controller(i2c, dir_name=res_dir)

    s_est = i2c.get_marginal_trajectory()
    x, y, z = env.run(policy_i2c)
    z_est = i2c.get_marginal_observed_trajectory()
    traj_eval.eval(z, z_est)

    lam_x3 = np.asarray([c.lambda_x3_b for c in i2c.cells]).squeeze() * i2c.alpha
    nu_x3 = np.asarray([c.nu_x3_b for c in i2c.cells]).squeeze() * i2c.alpha


    f, a = plt.subplots(2,1)
    _P = P.reshape((-1, env.dim_x * 2))
    _lam_x3 = lam_x3.reshape((-1, env.dim_x * 2))
    for i in range(env.dim_x * 2):
        a[0].plot(_P[:, i], 'k+-',
            label="$\mathbold{{P}}$" if i == 0 else "_")
        a[0].plot(_lam_x3[:, i], 'rx',\
            label="$\mathbold{{\Lambda}}_{{\overleftarrow{{x}}}}$" if i == 0 else "_")

    a[0].legend()
    a[0].set_title("Value Function parameters")
    a[0].set_ylabel("Quadratic Weights")
    a[1].set_ylabel("Linear Weights")
    a[1].set_xlabel("Timesteps")
    for i in range(env.dim_x):
        a[1].plot(p[:, i], 'k+-',
            label="$\mathbold{{p}}$" if i == 0 else "_")
        a[1].plot(-nu_x3[:, i], 'rx',
            label="$-\mathbold{{\\nu}}_{{\overleftarrow{{x}}}}$" if i == 0 else "_")
    a[1].legend()
    if res_dir is not None:
        plt.savefig(os.path.join(res_dir,
            "value.png"),
            bbox_inches='tight', format='png')
        matplotlib2tikz.save(
                os.path.join(res_dir, "value.tex"))
        plt.close(f)


if __name__ == "__main__":
    import matplotlib
    PLOT_TIKZ = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['figure.figsize'] = [10, 10]
    matplotlib.rcParams['legend.fontsize'] = 16
    matplotlib.rcParams['axes.titlesize'] = 22
    matplotlib.rcParams['axes.labelsize'] = 22
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}"]
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{fixmath}"]
    main()
    plt.show()