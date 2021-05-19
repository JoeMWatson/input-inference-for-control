"""
Common definition for environments and models
"""

import numpy as np
import numdifftools as nd
import logging

import i2c.env_autograd as dyn


class BaseDef(object):
    """Base definiton with several helpers."""

    def __init__(self, *args, **kwargs):
        self._pickle_args = args
        self._pickle_kwargs = kwargs
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        return {
            "_pickle_args": self._pickle_args,
            "_pickle_kwargs": self._pickle_kwargs,
        }

    def __setstate__(self, d):
        out = type(self)(*d["_pickle_args"], **d["_pickle_kwargs"])
        self.__dict__.update(out.__dict__)

    name = "Template"

    deterministic = False

    dim_x = None
    dim_xf = None  # feature transformation of state for learning
    dim_u = None
    xag = None
    x0 = None
    x0_dist = None
    xu_lim = None

    @property
    def random_starting_state(self):
        return self.x0_dist is not None

    @property
    def dim_s(self):
        return self.dim_x + self.dim_u

    @property
    def dim_xu(self):
        return self.dim_x + self.dim_u

    @property
    def dim_xt(self):
        return self.dim_xf + self.dim_u

    @property
    def dim_yt(self):
        return self.dim_x

    @property
    def dim_xat(self):
        return self.dim_xu

    def _zg(self):
        if self.xag is not None:
            return np.vstack((self.xag, np.zeros((self.dim_u, 1))))
        else:
            return np.zeros((self.dim_u, 1))

    @property
    def zg(self):
        return self._zg()

    @property
    def zg_term(self):
        return self.zg

    @property
    def zgc(self):
        return np.vstack((self.xg, np.zeros((self.dim_u, 1))))

    def observe(self, xu):
        raise NotImplementedError

    def observe_linearize(self, xu):
        raise NotImplementedError

    def observe_terminal(self, x):
        raise NotImplementedError

    def observe_terminal_linearize(self, x):
        raise NotImplementedError

    def remove_state_bounds(self):
        pass

    def xu_in_bounds(self, xu):
        assert self.xu_lim is not None
        assert self.xu_lim.shape == (2, self.dim_s)
        in_bounds = np.all(self.xu_lim[0, :, None] < xu) and np.all(
            self.xu_lim[1, :, None] > xu
        )
        return in_bounds

    def x_in_bounds(self, xu):
        x = xu[:, : self.dim_x]
        assert self.xu_lim is not None
        assert self.xu_lim.shape == (2, self.dim_s)
        in_bounds = np.all(self.xu_lim[0, : self.dim_x, None] < x) and np.all(
            self.xu_lim[1, : self.dim_x, None] > x
        )
        return in_bounds

    def clip_u(self, u):
        return np.clip(u, self.xu_lim[0, self.dim_x :], self.xu_lim[1, self.dim_x :])

    def filter_state_constraint_violations(self, xu, dx):
        assert self.xu_lim is not None
        assert self.xu_lim.shape == (2, self.dim_s)
        tol = 0.999
        xu_violate = np.clip(
            xu[:, : self.dim_x],
            tol * self.xu_lim[0, : self.dim_x],
            tol * self.xu_lim[1, : self.dim_x],
        )
        idx = np.any(xu_violate != xu[:, : self.dim_x], axis=1).nonzero()[0]
        if idx.size == 0:  # no violations
            return xu, dx
        else:
            first_idx = np.min(idx)
            logging.info(
                f"State limit violation detected. Cutting episode at timestep {first_idx}: {xu[first_idx, :]}"
            )
            return xu[:first_idx, :], dx[:first_idx, :]


class LinearDef(BaseDef):
    """LDS definiton. Used for LQR equivalance."""

    key = ["$x_1$", "$x_2$", "$u$"]
    z_key = key
    unit = [None, None, None]
    x_whiten = [True, True, True]
    y_whiten = [True, True]
    dim_x = 2
    dim_z = 3
    dim_u = 1
    dim_z_term = 2

    # training
    x_noise = np.diag([1e-3, 1e-3, 0.0])
    y_noise = np.diag([1e-3, 1e-3])

    x0 = np.array([[5.0, 5.0]]).T
    xg = np.array([[1.0, -1.0]]).T
    xag = xg
    zg_term = xg
    sig_x0 = 1e-20 * np.eye(dim_x)
    sig_eta = 1e-20 * np.eye(dim_x)

    xu_lim = np.array([[np.NINF, np.NINF, np.NINF], [np.Inf, np.Inf, np.Inf]])

    # system
    A = np.array([[1.1, 0.0], [0.1, 1.1]])
    a = (xg - A @ xg).reshape((dim_x, 1))
    B = np.array([[0.1], [0.0]])
    AB = np.concatenate((A, B), axis=1)

    C = np.vstack((np.eye(2), np.zeros((1, 2))))
    c = np.zeros((3, 1))
    D = np.array([[0.0, 0.0, 1.0]]).T

    def observe(self, xu):
        z = xu + self.c.T
        return z

    def observe_linearize(self, xu):
        z = self.observe(xu).T
        return z, self.C, self.c, self.D

    def observe_terminal(self, x):
        """z is (N, Dz)."""
        return x

    def observe_terminal_linearize(self, x):
        """z is (1, Dz)."""
        z = self.observe_terminal(x.T).T
        C = np.eye(self.dim_x)
        return z, C, self.c


class LinearMinimumEnergyDef(LinearDef):
    """LDS definiton for only control regularization. Used for covaraince control."""

    dim_z = 1

    x0 = np.array([[5.0, 5.0]]).T
    sig_x0 = np.diag([1e-1, 5e0])

    xag = None
    zg_term = np.array([[-5.0, -5.0]]).T

    A = np.array([[1.05, 0.0], [0.05, 1.01]])
    B = np.array([[0.1], [0.0]])
    a = (zg_term - A.dot(zg_term)).reshape((2, 1))
    AB = np.concatenate((A, B), axis=1)
    sig_eta = np.diag([1e-1, 1e-2])

    C = np.zeros((1, 2))
    c = np.zeros((1, 1))
    D = np.eye(1)

    C_terminal = np.eye(2)
    c_terminal = np.zeros((2, 1))
    D_terminal = np.zeros((2, 1))

    def observe(self, xu):
        return xu[:, self.dim_x :]

    def observe_linearize(self, xu):
        z = self.observe(xu).T
        return z, self.C, self.c, self.D

    def observe_terminal(self, x):
        return x

    def observe_terminal_linearize(self, x):
        return self.observe_terminal(x), self.C_terminal, self.c_terminal


class PendulumDef(BaseDef):
    """Pendulum definiton."""

    name = "Pendulum"
    key = ["$\\theta$", "$\\dot{\\theta}$", "$u$"]
    z_key = ["$\\sin(\\theta)$", "$\\cos(\\theta)$", "$\\dot{\\theta}$", "$u$"]
    unit = ["rad", "rad/s", "Nm"]

    dim_x = 2
    dim_xuf = 4
    dim_u = 1
    dim_z = 4
    dim_z_term = 3

    # states
    x0 = np.array([[np.pi, 0.0]]).T
    xg = np.array([[0.0, 0.0]]).T
    xag = np.array([[0.0, 1.0, 0.0]]).T
    zg_term = np.array([[0.0, 1.0, 0.0]]).T

    sig_x0 = 1e-5 * np.eye(dim_x)
    # system
    D = np.array([[0.0, 0.0, 0.0, 1.0]]).T  # constant

    sig_eta = np.diag([1e-5, 1e-5])

    xu_lim = np.array([[np.NINF, np.NINF, -2.0], [np.Inf, np.Inf, 2.0]])

    @staticmethod
    def featurespace(xu):
        return np.stack(
            (
                np.sin(xu[:, 0]),
                np.cos(xu[:, 0]),
                xu[:, 1],
                xu[:, 2],
            ),
            axis=1,
        )

    def observe(self, xu):
        """z is (N, Dz)."""
        z = np.vstack((np.sin(xu[:, 0]), np.cos(xu[:, 0]), xu[:, 1], xu[:, 2])).T
        return z

    def observe_linearize(self, xu):
        """z is (1, Dz)."""
        z = self.observe(xu).T
        x, u = xu[:, : self.dim_x], xu[:, self.dim_x :]
        C = np.array(
            [[np.cos(xu[0, 0]), 0.0], [-np.sin(xu[0, 0]), 0.0], [0.0, 1.0], [0.0, 0.0]]
        )
        c = z - C @ x.T - self.D @ u.T
        return z, C, c, self.D

    def observe_terminal(self, x):
        """z is (N, Dz)."""
        z = np.vstack((np.sin(x[:, 0]), np.cos(x[:, 0]), x[:, 1])).T
        return z

    def observe_terminal_linearize(self, x):
        """x is (Dz, 1)."""
        z = self.observe_terminal(x.T).T
        C = np.array([[np.cos(x[0, 0]), 0.0], [-np.sin(x[0, 0]), 0.0], [0.0, 1.0]])
        c = z - C @ x
        return z, C, c


class PendulumKnown(PendulumDef):
    """Pendulum definiton."""

    @staticmethod
    def dynamics(xu):
        return dyn.pendulum_dynamics(xu)

    def dydxu(self, xu):
        return dyn.pendulum_dydxu(xu).reshape((self.dim_x, self.dim_xu))


class PendulumKnownActReg(PendulumKnown):
    """Pendulum definiton for only control regularization. Used for covaraince control."""

    z_key = ["$u$"]
    dim_xa = 0  # augmented for 'observation'
    xg = np.array([[None, None]]).T
    xag = None

    name = "Pendulum"
    key = ["$\\theta$", "$\\dot{\\theta}$", "$u$"]
    unit = ["rad", "rad/s", "Nm"]

    dim_x = 2
    dim_xuf = 4  #
    dim_u = 1
    dim_z = 1
    dim_z_term = 1

    def observe(self, xu):
        """z is (N, Dz)."""
        return xu[:, self.dim_x :]

    def observe_linearize(self, xu):
        """z is (1, Dz)."""
        z = self.observe(xu).T

        C = np.zeros((self.dim_z, self.dim_x))
        c = z - self.D @ xu[:, self.dim_x :].T
        return z, C, c, self.D

    def observe_terminal(self, x):
        return None

    def observe_terminal_linearize(self, x):
        return None, None, None


class FurutaDef(BaseDef):
    """Furuta Pendulum definiton."""

    name = "Furuta"
    key = ["$\\theta$", "$\\phi$", "$\\dot{\\theta}$", "$\\dot{\\phi}$", "$u$"]
    z_key = [
        "$\\theta$",
        "$\\sin(\\phi)$",
        "$\\cos(\\phi)$",
        "$\\dot{\\theta}$",
        "$\\dot{\\phi}$",
        "$u$",
    ]
    unit = ["rad", "rad", "rad/s", "rad/s", "Nm"]
    dim_x = 4
    dim_u = 1
    dim_z = 6
    dim_z_term = 5
    dim_xt = 7

    # states
    x0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T
    xg = np.array([[0.0, np.pi, 0.0, 0.0]]).T
    xag = np.array(
        [
            [
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
            ]
        ]
    ).T
    zg_term = np.array([[0.0, 0.0, -1.0, 0.0, 0.0]]).T

    sig_x0 = 1e-6 * np.eye(dim_x)
    # sig_eta = np.diag([1e-10, 1e-10, 1e-10, 1e-10])
    sig_eta = np.diag([1e-7, 1e-7, 1e-7, 1e-7])

    # system
    D = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]).T  # constant

    xt_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    xt_std = np.array([1.5, 1.0, 1.0, 1.0, 3.0, 10.0, 5.0])
    yt_mean = np.zeros((dim_x,))
    yt_std = 0.1 * np.ones((dim_x,))

    # data augmentation
    x_noise = np.diag([1e-9, 1e-9, 1e-9, 1e-9, 1e-9])
    y_noise = np.diag([1e-6, 1e-6, 1e-6, 1e-6])

    xu_lim = np.array(
        [[-2, np.NINF, np.NINF, np.NINF, -5.0], [2, np.Inf, np.Inf, np.Inf, 5.0]]
    )

    dynamics_lin = nd.Jacobian(lambda s: dyn.furuta_dynamics(s[None, :4], s[None, 4:]))

    @classmethod
    def dydxu(cls, x, u):
        xu = np.hstack((x, u))[0, :]
        AB = cls.dynamics_lin(xu)
        return AB[:, : cls.dim_x]

    @classmethod
    def dydu(cls, x, u):
        xu = np.hstack((x, u))[0, :]
        AB = cls.dynamics_lin(xu)
        return AB[:, cls.dim_x :]

    @staticmethod
    def featurespace(xu):
        return np.stack(
            (
                np.sin(xu[:, 0]),
                np.cos(xu[:, 0]),
                np.sin(xu[:, 1]),
                np.cos(xu[:, 1]),
                xu[:, 2],
                xu[:, 3],
                xu[:, 4],
            ),
            1,
        )

    def observe(self, xu):
        return np.vstack(
            [
                [
                    xu[:, 0],
                    np.sin(xu[:, 1]),
                    np.cos(xu[:, 1]),
                    xu[:, 2],
                    xu[:, 3],
                    xu[:, 4],
                ]
            ]
        ).T

    def observe_linearize(self, xu):
        z = self.observe(xu).T
        A = np.zeros((self.dim_z, self.dim_x))
        B = np.zeros((self.dim_z, self.dim_u))
        B[-1, -1] = 1.0
        A[[0, 3, 4], [0, 2, 3]] = 1.0
        A[1, 1] = np.cos(xu[1, :].T)
        A[2, 1] = np.cos(xu[1, :].T)
        x, u = xu[:, : self.dim_x], xu[:, self.dim_x :]
        a = z - A @ x - B @ u
        return z, A, B, a

    def observe_terminal(self, x):
        return np.vstack(
            [
                [
                    x[:, 0],
                    np.sin(x[:, 1]),
                    np.cos(x[:, 1]),
                    x[:, 2],
                    x[:, 3],
                ]
            ]
        ).T

    def observe_terminal_linearize(self, x):
        z = self.observe_terminal(x.T).T
        A = np.zeros((self.dim_z_term, self.dim_x))
        A[[0, 3, 4], [0, 2, 3]] = 1.0
        A[1, 1] = np.cos(x[1, :].T)
        A[2, 1] = np.cos(x[1, :].T)
        a = z - A @ x
        return z, A, a


class FurutaKnown(FurutaDef):
    """Furuta Pendulum definiton."""

    @staticmethod
    def dynamics(x, u):
        return dyn.furuta_dynamics(x, u)


class BaseCartpoleDef(BaseDef):
    """Base Cartpole definiton."""

    name = "Cartpole"
    key = ["$x$", "$\\theta$", "$\\dot{x}$", "$\\dot{\\theta}$", "$u$"]
    z_key = [
        "$x$",
        "$\\sin(\\theta)$",
        "$\\cos(\\theta)$",
        "$\\dot{x}$",
        "$\\dot{\\theta}$",
        "$u$",
    ]
    unit = ["m", "rad", "m/s", "rad/s", "Nm"]
    # environmental data scaling parameters
    x_whiten = [True, False, True, True, True]
    y_whiten = [True, True, True, True]
    # dimensions
    dim_x = 4
    dim_xuf = 6
    dim_z = 6
    dim_z_term = 5
    dim_u = 1
    # states
    sig_x0 = 1e-5 * np.eye(dim_x)
    # system
    D = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T  # constant
    sig_eta = np.diag([1e-5, 1e-5, 5e-4, 5e-4])
    # data augmentation
    x_noise = np.diag([1e-9, 1e-9, 1e-9, 1e-9, 1e-9])
    y_noise = np.diag([1e-6, 1e-6, 1e-6, 1e-6])

    @staticmethod
    def featurespace(xu):
        return np.stack(
            (
                xu[:, 0],
                np.sin(xu[:, 1]),
                np.cos(xu[:, 1]),
                xu[:, 2],
                xu[:, 3],
                xu[:, 4],
            ),
            1,
        )

    def observe(self, xu):
        return np.vstack(
            [
                [
                    xu[:, 0],
                    np.sin(xu[:, 1]),
                    np.cos(xu[:, 1]),
                    xu[:, 2],
                    xu[:, 3],
                    xu[:, 4],
                ]
            ]
        ).T

    def observe_linearize(self, xu):
        y = self.observe(xu).T
        C = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, np.cos(xu[0, 1]), 0.0, 0.0],
                [0.0, -np.sin(xu[0, 1]), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        x, u = xu[:, : self.dim_x].T, xu[:, self.dim_x :].T
        c = y - C @ x - self.D @ u
        return y, C, c, self.D

    def observe_terminal(self, x):
        """z is (N, Dz)."""
        z = np.vstack([[x[:, 0], np.sin(x[:, 1]), np.cos(x[:, 1]), x[:, 2], x[:, 3]]]).T
        return z

    def observe_terminal_linearize(self, x):
        """z is (1, Dz)."""
        z = self.observe_terminal(x.T).T
        C = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, np.cos(x[1, 0]), 0.0, 0.0],
                [0.0, -np.sin(x[1, 0]), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        c = z - C.dot(x)
        return z, C, c


class CartpoleDef(BaseCartpoleDef):

    x0 = np.array([[0.0, np.pi, 0.0, 0.0]]).T
    xg = np.array([[0.0, 0.0, 0.0, 0.0]]).T
    xag = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]).T
    zg_term = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]).T
    sig_eta = np.diag([1e-8, 1e-8, 1e-8, 1e-8])

    xu_lim = np.array(
        [
            [np.NINF, np.NINF, np.NINF, np.NINF, -5.0],
            [np.Inf, np.Inf, np.Inf, np.Inf, 5.0],
        ]
    )


class CartpoleKnown(CartpoleDef):
    """Cartpole definiton."""

    @staticmethod
    def dynamics(xu):
        return dyn.cartpole_dynamics(xu)

    def dydxu(self, xu):
        return dyn.cartpole_dydxu(xu).reshape((self.dim_x, self.dim_xu))


class DoubleCartpoleDef(BaseDef):
    """Double Cartpole definiton."""

    name = "Double Cartpole"
    key = [
        "$x$",
        "$\\theta_1$",
        "$\\theta_2$",
        "$\\dot{x}$",
        "$\\dot{\\theta}_1$",
        "$\\dot{\\theta}_2$",
        "$u$",
    ]
    z_key = [
        "$x$",
        "$\\sin\\theta_1$",
        "$\\cos\\theta_1$",
        "$\\sin\\theta_2$",
        "$\\cos\\theta_2$",
        "$\\dot{x}$",
        "$\\dot{\\theta}_1$",
        "$\\dot{\\theta}_2$",
        "$u$",
    ]
    unit = ["m", "rad", "rad", "m/s", "rad/s", "rad/s", "Nm"]
    x_whiten = [True, False, False, True, True, True, True]
    y_whiten = [True, False, False, True, True, True]
    dim_x = 6
    dim_xuf = 9
    dim_z = 9
    dim_u = 1
    dim_z_term = 8
    # states
    x0 = np.array([[0.0, np.pi, np.pi, 0.0, 0.0, 0.0]]).T
    xg = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    xag = np.array([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T
    zg = np.vstack((xag, np.zeros((dim_u, dim_u))))
    zgc = np.vstack((xg, np.zeros((dim_u, dim_u))))
    zg_term = xag
    sig_x0 = 1e-6 * np.eye(dim_x)
    D = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T  # constant
    sig_eta = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])

    xu_lim = np.array(
        [
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, -10.0],
            [np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 10.0],
        ]
    )

    @staticmethod
    def featurespace(xu):
        return np.stack(
            (
                xu[:, 0],
                np.sin(xu[:, 1]),
                np.cos(xu[:, 1]),
                np.sin(xu[:, 2]),
                np.cos(xu[:, 2]),
                xu[:, 3],
                xu[:, 4],
                xu[:, 5],
                xu[:, 6],
            ),
            1,
        )

    def observe(self, xu):
        return np.vstack(
            (
                xu[:, 0],
                np.sin(xu[:, 1]),
                np.cos(xu[:, 1]),
                np.sin(xu[:, 2]),
                np.cos(xu[:, 2]),
                xu[:, 3],
                xu[:, 4],
                xu[:, 5],
                xu[:, 6],
            )
        ).T

    def observe_linearize(self, xu):
        assert xu.shape == (1, self.dim_xu)
        y = self.observe(xu).T

        C = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, np.cos(xu[0, 1]), 0.0, 0.0, 0.0, 0.0],
                [0.0, -np.sin(xu[0, 1]), 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, np.cos(xu[0, 2]), 0.0, 0.0, 0.0],
                [0.0, 0.0, -np.sin(xu[0, 2]), 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        x, u = xu[:, : self.dim_x].T, xu[:, self.dim_x :].T
        c = y - C @ x - self.D @ u
        return y, C, c, self.D

    def observe_terminal(self, x):
        """z is (N, Dz)."""
        return np.vstack(
            (
                x[:, 0],
                np.sin(x[:, 1]),
                np.cos(x[:, 1]),
                np.sin(x[:, 2]),
                np.cos(x[:, 2]),
                x[:, 3],
                x[:, 4],
                x[:, 5],
            )
        ).T

    def observe_terminal_linearize(self, x):
        """z is (1, Dz)."""
        z = self.observe_terminal(x.T).T
        C = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, np.cos(x[1, 0]), 0.0, 0.0, 0.0, 0.0],
                [0.0, -np.sin(x[1, 0]), 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, np.cos(x[2, 0]), 0.0, 0.0, 0.0],
                [0.0, 0.0, -np.sin(x[2, 0]), 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        c = z - C @ x
        return z, C, c


class DoubleCartpoleKnown(DoubleCartpoleDef):
    """Double Cartpole definiton."""

    @staticmethod
    def dynamics(xu):
        return dyn.double_cartpole_dynamics(xu)

    def dydxu(self, xu):
        return dyn.double_cartpole_dydxu(xu).reshape((self.dim_x, self.dim_xu))
