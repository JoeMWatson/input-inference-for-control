""""
Make some nice gifs for talks and the README
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
from pygifsicle import optimize
from tqdm import tqdm

from i2c.env import make_env
from i2c.i2c import I2cGraph
from i2c.model import make_env_model

DIR_NAME = os.path.dirname(__file__)


def pi_format(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)


def make_dcp_trajopt_gif():
    from experiments import double_cartpole_known_cq as experiment

    gif_filename = os.path.join(DIR_NAME, "..", "assets", "dcp_%ds.gif")

    stream = []
    model = make_env_model(experiment.ENVIRONMENT, experiment.MODEL)
    i2c = I2cGraph(
        model,
        experiment.N_DURATION,
        experiment.INFERENCE.Q,
        experiment.INFERENCE.R,
        experiment.INFERENCE.Qf,
        experiment.INFERENCE.alpha,
        experiment.INFERENCE.alpha_update_tol,
        # experiment.INFERENCE.mu_u,
        1e-8 * np.random.randn(experiment.N_DURATION, 1),
        0.75 * np.eye(1),
        # experiment.INFERENCE.sig_u,
        experiment.INFERENCE.mu_x_term,
        experiment.INFERENCE.sig_x_term,
        experiment.INFERENCE.inference,
        res_dir=None,
    )
    time = range(experiment.N_DURATION)
    opacity = 0.3
    L = 0.3365
    for i in tqdm(range(experiment.N_INFERENCE)):

        i2c.learn_msgs()

        if i % 1 == 0:
            fig = plt.figure()

            gs = fig.add_gridspec(3, 2)
            ax_x1 = fig.add_subplot(gs[0, 0])
            ax_x2 = fig.add_subplot(gs[1, 0])
            ax_u = fig.add_subplot(gs[2, 0])
            ax_xy = fig.add_subplot(gs[:, 1])

            ax_x1.set_title("Double Cartpole\nTrajectory Optimizaiton")
            ax_xy.set_title(f"Iteration {i:03d}")
            ax_x1.set_ylabel("$\\theta_0$")
            ax_x2.set_ylabel("$\\theta_1$")
            ax_xy.set_ylabel("$y$")
            ax_xy.set_xlabel("$x$")
            ax_u.set_ylabel("$u$")
            ax_u.set_xlabel("$n$")

            mu_xu, sig_xu = i2c.get_marginal_state_action_distribution()
            for d, ax in zip([1, 2, -1], [ax_x1, ax_x2, ax_u]):
                xp_u, xp_l = i2c.indexed_confidence_bound(mu_xu, sig_xu, d)
                ax.fill_between(
                    time, xp_l, xp_u, where=xp_u >= xp_l, facecolor="c", alpha=opacity
                )
                ax.plot(time, mu_xu[:, d], "c")
                ax.plot(time, np.zeros((experiment.N_DURATION,)), "k--")

            x_tip = mu_xu[:, 0] + L * np.sin(mu_xu[:, 1]) + L * np.sin(mu_xu[:, 2])
            y_tip = L * np.cos(mu_xu[:, 1]) + L * np.cos(mu_xu[:, 2])
            T = mu_xu.shape[0]
            for t in range(T):
                x0 = mu_xu[t, 0]
                x1 = x0 + L * np.sin(mu_xu[t, 1])
                x2 = x1 + L * np.sin(mu_xu[t, 2])
                y0 = 0
                y1 = y0 + L * np.cos(mu_xu[t, 1])
                y2 = y1 + L * np.cos(mu_xu[t, 2])
                ax_xy.plot([x0, x1], [y0, y1], color="b", alpha=0.2 * t / T)
                ax_xy.plot([x1, x2], [y1, y2], color="b", alpha=0.2 * t / T)
            ax_xy.plot(x_tip, y_tip, color="b", alpha=0.5)
            ax_xy.plot(0, -2 * L, "kx", markersize=10)
            ax_xy.plot(
                np.linspace(-1.5, 1.5, 100),
                2
                * L
                * np.ones(
                    100,
                ),
                "k--",
            )

            ax_x1.set_ylim(-np.pi, 3 * np.pi)
            ax_x2.set_ylim(-np.pi, 3 * np.pi)
            ax_u.set_ylim(-10, 10)
            ax_x1.set_xticks([])
            ax_x2.set_xticks([])
            ax_xy.set_xticks([])
            ax_xy.set_yticks([])

            ax_x1.yaxis.set_major_formatter(plt.FuncFormatter(pi_format))
            ax_x2.yaxis.set_major_formatter(plt.FuncFormatter(pi_format))
            for a in [ax_x1, ax_x2, ax_u]:
                a.set_xlim(0, experiment.N_DURATION)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            stream.append(image.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
            # plt.show()
            plt.close(fig)

        # kwargs_write = {'fps': 30.0, 'quantizer': 'nq'}
        for T in [1, 2, 3, 4, 5, 10]:
            name = gif_filename % T
            fps = len(stream) / T
            imageio.mimsave(name, stream, fps=fps)
            optimize(name)


def make_pendulum_cov_control_gif():
    import experiments.pendulum_known_act_reg_quad as experiment
    from i2c.policy.linear import TimeIndexedLinearGaussianPolicy
    from i2c.utils import covariance_2d

    model = make_env_model(experiment.ENVIRONMENT, experiment.MODEL)
    env = make_env(experiment)

    i2c = I2cGraph(
        sys=model,
        horizon=experiment.N_DURATION,
        Q=experiment.INFERENCE.Q,
        R=experiment.INFERENCE.R,
        Qf=experiment.INFERENCE.Qf,
        alpha=experiment.INFERENCE.alpha,
        alpha_update_tol=experiment.INFERENCE.alpha_update_tol,
        mu_u=experiment.INFERENCE.mu_u,
        # sig_u=experiment.INFERENCE.sig_u,
        sig_u=1.0 * np.eye(1),
        mu_x_terminal=experiment.INFERENCE.mu_x_term,
        sig_x_terminal=experiment.INFERENCE.sig_x_term,
        inference=experiment.INFERENCE.inference,
        res_dir=None,
    )
    for c in i2c.cells:
        c.use_expert_controller = False

    policy = TimeIndexedLinearGaussianPolicy(
        experiment.POLICY_COVAR, experiment.N_DURATION, i2c.sys.dim_u, i2c.sys.dim_x
    )

    i2c._propagate = False
    experiment.N_INFERENCE = 200
    iters = range(experiment.N_INFERENCE)
    gif_filename = os.path.join(DIR_NAME, "..", "assets", "p_cc_%ds.gif")
    stream = []
    for iter in tqdm(iters):
        i2c.learn_msgs()
        policy.write(*i2c.get_local_linear_policy())
        xs, _, _, _ = env.batch_eval(policy=policy, n_eval=500, deterministic=False)
        fig, ax = plt.subplots(1, 1)
        a = ax
        a.set_title(f"Pendulum Covariance Control\nIteration {iter:03d}")

        for i, x in enumerate(xs):
            a.plot(x[:, 0], x[:, 1], ".c", alpha=0.1, markersize=1)
            a.plot(
                x[-1, 0],
                x[-1, 1],
                ".c",
                alpha=1.0,
                label="rollouts" if i == 0 else None,
                markersize=1,
            )

        covariance_2d(i2c.sys.sig_x0, i2c.sys.x0, a, facecolor="k")
        a.plot(
            i2c.sys.x0[0], i2c.sys.x0[1], "xk", label="$\\mathbf{x}_0$", markersize=3
        )
        covariance_2d(i2c.sig_x_terminal, i2c.mu_x_terminal, a, facecolor="r")
        a.plot(
            i2c.mu_x_terminal[0],
            i2c.mu_x_terminal[1],
            "xr",
            label="$\\mathbf{x}_g$",
            markersize=3,
        )

        a.set_xlabel(i2c.sys.key[0])
        a.set_ylabel(i2c.sys.key[1])
        a.set_xlim(-np.pi / 4, 3 * np.pi / 2)
        a.set_ylim(-5, 5)
        a.legend(loc="lower left")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        stream.append(image.reshape(fig.canvas.get_width_height()[::-1] + (3,)))

        plt.close(fig)
    for T in [1, 2, 3, 4, 5, 10]:
        name = gif_filename % T
        fps = len(stream) / T
        imageio.mimsave(name, stream, fps=fps)
        optimize(name)


if __name__ == "__main__":
    import matplotlib

    matplotlib.rcParams["font.family"] = "serif"
    # make_dcp_trajopt_gif()
    make_pendulum_cov_control_gif()
