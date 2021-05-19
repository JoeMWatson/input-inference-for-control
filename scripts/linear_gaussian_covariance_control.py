"""
Linear Gaussian Covariance Control with Input Inference for Control
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz
from tqdm import tqdm

from i2c.i2c import I2cGraph
from i2c.env import make_env
from i2c.model import make_env_model

from i2c.policy.linear import TimeIndexedLinearGaussianPolicy
from i2c.utils import covariance_2d, make_results_folder

import scripts.experiments.linear_known_covariance_control as experiment

DIR_NAME = os.path.dirname(__file__)


def plot_covariance_control(i2c, xs, filename="", dir_name=None):
    f, ax = plt.subplots(2, 1)
    a = ax[0]
    a.set_title("Linear Gaussian Covariance Control")

    for i, x in enumerate(xs):
        a.plot(x[:, 0], x[:, 1], "--.c", alpha=0.5, markersize=1)
        a.plot(
            x[-1, 0],
            x[-1, 1],
            ".c",
            alpha=1.0,
            label="rollouts" if i == 0 else None,
            markersize=1,
        )

    covariance_2d(i2c.sys.sig_x0, i2c.sys.x0, a, facecolor="k")
    a.plot(i2c.sys.x0[0], i2c.sys.x0[1], "xk", label="$\\mathbf{x}_0$", markersize=3)
    covariance_2d(i2c.sig_x_terminal, i2c.mu_x_terminal, a, facecolor="r")
    a.plot(
        i2c.mu_x_terminal[0],
        i2c.mu_x_terminal[1],
        "xr",
        label="$\\mathbf{x}_g$",
        markersize=3,
    )

    for i, c in enumerate(i2c.cells):
        a.plot(c.mu_x0_pf[0], c.mu_x0_pf[1], ".-g", markersize=1)
        opac = i / len(i2c.cells)
        covariance_2d(
            c.sig_x0_pf, c.mu_x0_pf, a, facecolor="g", linestyle="--", alpha=opac
        )

    c = i2c.cells[-1]
    a.plot(c.mu_x3_pf[0], c.mu_x3_pf[1], ".-g", label="closed-loop", markersize=1)
    covariance_2d(c.sig_x3_pf, c.mu_x3_pf, a, facecolor="g", linestyle="--")

    # check values
    print(f"goal mean {i2c.mu_x_terminal}")
    print(f"posterior mean {c.mu_x3_m}")
    print(f"propagated mean {c.mu_x3_pf}")

    print(f"goal covariance {i2c.sig_x_terminal}")
    print(f"posterior covariance {c.sig_x3_m}")
    print(f"propagated covariance {c.sig_x3_pf}")

    a.set_xlabel(i2c.sys.key[0])
    a.set_ylabel(i2c.sys.key[1])
    a.legend(loc="upper left")

    a = ax[1]
    kl_term = np.asarray(i2c.kl_terms)
    a.plot(kl_term, "kx-", markersize=1)
    a.set_yscale("log")
    a.set_ylabel("KL$(\\mathbf{x}_T||\\mathbf{x}^*_T)$")
    a.set_xlabel("Iterations")

    if dir_name is not None:
        plt.savefig(
            os.path.join(dir_name, "covariance_control_{}.png".format(filename)),
            bbox_inches="tight",
            format="png",
        )
        matplotlib2tikz.save(
            os.path.join(dir_name, "covariance_control_{}.tex".format(filename))
        )


def main():
    res_dir = make_results_folder("i2c_linear_covariance_control", 0, "", release=True)
    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT, experiment.MODEL)

    i2c = I2cGraph(
        sys=model,
        horizon=experiment.N_DURATION,
        Q=experiment.INFERENCE.Q,
        R=experiment.INFERENCE.R,
        Qf=experiment.INFERENCE.Qf,
        alpha=experiment.INFERENCE.alpha,
        alpha_update_tol=experiment.INFERENCE.alpha_update_tol,
        mu_u=experiment.INFERENCE.mu_u,
        sig_u=experiment.INFERENCE.sig_u,
        mu_x_terminal=experiment.INFERENCE.mu_x_term,
        sig_x_terminal=experiment.INFERENCE.sig_x_term,
        inference=experiment.INFERENCE.inference,
        res_dir=res_dir,
    )
    for c in i2c.cells:
        c.use_expert_controller = False
    i2c._propagate = True

    policy = TimeIndexedLinearGaussianPolicy(
        experiment.POLICY_COVAR, experiment.N_DURATION, i2c.sys.dim_u, i2c.sys.dim_x
    )

    for i in tqdm(range(experiment.N_INFERENCE)):
        i2c.learn_msgs()

    policy.write(*i2c.get_local_linear_policy())

    xs, _, _, _ = env.batch_eval(policy=policy, n_eval=100, deterministic=False)
    plot_covariance_control(i2c, xs, filename="", dir_name=res_dir)


if __name__ == "__main__":
    np.random.seed(0)
    main()
    plt.show()
