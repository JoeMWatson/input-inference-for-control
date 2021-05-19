import numpy as np
import logging

from i2c.exp_types import CubatureQuadrature, GaussHermiteQuadrature


class QuadratureInference(object):
    def __init__(self, params, dim):
        assert isinstance(params, (CubatureQuadrature, GaussHermiteQuadrature))
        self.dim = dim
        self.base_pts = params.pts(dim)
        self.sf, self.weights_mu, self.weights_sig = params.weights(dim)
        self.n_points = self.base_pts.shape[0]

    def get_x_pts(self, m_x, sig_x):
        m_x = m_x.reshape((-1, self.dim))
        try:
            L = np.linalg.cholesky(sig_x)
            scale = self.sf * L
        except:
            logging.exception(
                f"Bad Choleksy\nCov:\n{sig_x}\nEigvals:\n{np.linalg.eig(sig_x)}"
            )
            raise
        return m_x + self.base_pts @ scale.T

    def forward(self, f, m_x, sig_x):
        self.x_pts = self.get_x_pts(m_x, sig_x)
        self.y_pts, self.m_y, self.sig_y, self.sig_xy = self.forward_pts(
            f, m_x, self.x_pts
        )
        return self.m_y.T, self.sig_y

    def forward_pts(self, f, m_x, x_pts):
        y_pts = f(x_pts)
        m_y = (y_pts.T @ self.weights_sig).reshape((1, -1))
        # batched weighted outer products
        sig_y = np.einsum("b,bi,bj->ij", self.weights_sig, y_pts, y_pts) - np.outer(
            m_y, m_y
        )
        sig_xy = np.einsum("b,bi,bj->ij", self.weights_sig, x_pts, y_pts) - np.outer(
            m_x, m_y
        )
        return y_pts, m_y, sig_y, sig_xy

    def forward_gaussian(self, f, m_x, sig_x):
        self.x_pts = self.get_x_pts(m_x, sig_x)
        self.y_pts, self._sig_y = f(self.x_pts)
        self.m_y = np.einsum("b, bi->i", self.weights_sig, self.y_pts).reshape((1, -1))
        # batched weighted outer products
        self.sig_y = np.einsum(
            "b,bi,bj->ij", self.weights_sig, self.y_pts, self.y_pts
        ) - np.outer(self.m_y, self.m_y)
        self.sig_xy = np.einsum(
            "b,bi,bj->ij", self.weights_sig, self.x_pts, self.y_pts
        ) - np.outer(m_x, self.m_y)
        self.sig_noise = np.einsum("b,bij->ij", self.weights_sig, self._sig_y)
        return self.m_y.T, self.sig_y, self.sig_noise


if __name__ == """__main__""":
    import matplotlib.pyplot as plt
    import matplotlib2tikz
    from matplotlib.patches import Ellipse
    import numdifftools as nd

    def plot_ellipse(axis, mean, cov, n_std=2, facecolor="b"):
        w, v = np.linalg.eig(cov)
        _w = 2 * n_std * np.sqrt(w)  # diameter not radius
        assert not np.any(np.isnan(_w))  # fails silently otherwise

        theta = np.rad2deg(-np.arctan2(v[0, 1], v[0, 0]))
        ellipse = Ellipse(
            xy=mean.squeeze(),
            width=_w[0],
            height=_w[1],
            angle=theta,
            edgecolor=facecolor,
            facecolor="none",
        )
        axis.add_patch(ellipse)

    n_samp = 5000
    n_plot = 500

    fig, a = plt.subplots(1, 2)

    mean = np.zeros((2,))
    th = np.pi / 4
    T = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cov = T @ np.diag([0.5, 0.05]) @ T.T

    print(np.linalg.cholesky(cov))
    A, B = np.linalg.eig(cov)
    print(np.sqrt(np.diag(A)) @ B, A, B)

    def func(x):
        return np.concatenate(
            (
                np.sin(1.5 * x[:, 1, None] + 1) + 0.1 * x[:, 0, None],
                np.cos(1.5 * x[:, 1, None] + 1) + 0.1 * x[:, 0, None],
            ),
            axis=1,
        )

    new_func = lambda x: func(x[None, :])
    dfunc = nd.Jacobian(new_func)

    samp = np.random.randn(n_samp, 2)
    x_samp = np.random.multivariate_normal(mean, cov, n_samp).reshape((n_samp, 2))
    y_samp = func(x_samp)
    a[0].plot(x_samp[:n_plot, 0], x_samp[:n_plot, 1], "c.", alpha=1, markersize=1)
    a[1].plot(y_samp[:n_plot, 0], y_samp[:n_plot, 1], "c.", alpha=1, markersize=1)

    plot_ellipse(a[0], mean, cov, facecolor="k")

    ex_mean = func(mean.reshape((1, -1)))[0, :]
    J = dfunc(mean)
    ex_cov = J @ cov @ J.T
    a[1].plot(ex_mean[0], ex_mean[1], "m+", label="Linearize")
    plot_ellipse(a[1], ex_mean, ex_cov, facecolor="m")

    cub_inf = QuadratureInference(CubatureQuadrature(1, 0, 0), 2)
    quad_mean, quad_cov = cub_inf.forward(func, mean[:, None], cov)
    pts, y_pts = cub_inf.x_pts, cub_inf.y_pts
    a[0].plot(pts[1:, 0], pts[1:, 1], "bx", label="Cubature Points")
    a[1].plot(y_pts[1:, 0], y_pts[1:, 1], "bx")

    a[1].plot(quad_mean[0], quad_mean[1], "b+", label="Cubature")
    plot_ellipse(a[1], quad_mean, quad_cov, facecolor="b")

    cub_inf = QuadratureInference(GaussHermiteQuadrature(4), 2)
    quad_mean, quad_cov = cub_inf.forward(func, mean[:, None], cov)
    pts, y_pts = cub_inf.x_pts, cub_inf.y_pts
    a[0].plot(pts[:, 0], pts[:, 1], "yx", label="Gauss-Hermite Points")
    a[1].plot(y_pts[:, 0], y_pts[:, 1], "yx")

    a[1].plot(quad_mean[0], quad_mean[1], "y+", label="Gauss-Hermite")
    plot_ellipse(a[1], quad_mean, quad_cov, facecolor="y")

    mc_mean = np.mean(y_samp, axis=0)
    mc_cov = np.cov(y_samp.T)
    a[1].plot(mc_mean[0], mc_mean[1], "g+", label="Sample")
    plot_ellipse(a[1], mc_mean, mc_cov, facecolor="g")
    a[1].legend()

    a[0].set_xlim(-2, 2)
    a[0].set_ylim(-2, 2)

    matplotlib2tikz.save("inference.tex")

    plt.show()
