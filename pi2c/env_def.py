"""
Common definition for environments and models
"""

import numpy as np
import pi2c.env_autograd as dyn

class BaseDef(object):

    dim_x = None
    dim_xa = None
    dim_u = None
    xag = None
    xg = None

    @property
    def dim_y(self):
        return self.dim_xa + self.dim_u

    @property
    def dim_s(self):
        return self.dim_x + self.dim_u

    @property
    def dim_sa(self):
        return self.dim_xa + self.dim_u

    @property
    def dim_xt(self):
        return self.dim_x + self.dim_u

    @property
    def dim_yt(self):
        return self.dim_x

    @property
    def dim_xat(self):
        return self.dim_sa

    @property
    def sg(self):
        return np.vstack((self.xag, np.zeros((self.dim_u, self.dim_u))))

    @property
    def sgc(self):
        return np.vstack((self.xg, np.zeros((self.dim_u, self.dim_u))))


class LinearDef(BaseDef):

    key = ["$x_1$", "$x_2$", "$u$"]
    y_key = key
    unit = [None, None, None]
    x_whiten = [True, True, True]
    y_whiten = [True, True]
    dim_x = 2
    dim_xa = 2
    dim_u = 1

    x_noise = np.diag([1e-3, 1e-3, 0.0])
    y_noise = np.diag([1e-3, 1e-3])

    x0 = np.array([[5.0, 5.0]]).T
    xg = np.array([[0.0, 0.0]]).T
    xag = xg

    sigX0 = 1e-20 * np.eye(dim_x)
    sigV = 1e-20 * np.eye(dim_x)
    # sigV = 1e3 * np.eye(dim_x)

    # system
    A = np.array([
        [1.1, 0.],
        [0.1, 1.1]])
    a = np.zeros((dim_x,1))
    B = np.array([
        [0.1],
        [0.0]])
    C = np.vstack((np.eye(2), np.zeros((1, 2))))
    c = np.zeros((3, 1))
    D = np.array([[0.0, 0.0, 1.0]]).T

    def observe(self, x, u):
        y = self.C.dot(x) + self.D.dot(u)

        return y, self.C, self.c, self.D

class PendulumDef(BaseDef):

    key = ["$\\theta$", "$\dot{\\theta}$", "$u$"]
    y_key = ["$\sin(\\theta)$", "$\cos(\\theta)$", "$\dot{\\theta}$", "$u$"]
    unit = ["rad", "rad/s", "Nm"]
    x_whiten = [False, True, True]
    y_whiten = [True, True]
    dim_x = 2
    dim_xa = 3  # augmented for 'observation'
    dim_u = 1

    x_noise = np.diag([1e-6, 1e-6, 0.0])
    y_noise = np.diag([1e-4, 1e-4])
    # x_noise = np.diag([1e-5, 1e-5, 0.0])
    # y_noise = np.diag([1e-5, 1e-5])
    # states
    x0 = np.array([[np.pi, 0.0]]).T
    xg = np.array([[0.0, 0.0]]).T
    xag = np.array([[0.0, 1.0, 0.0]]).T
    sigX0 = 1e-20 * np.eye(dim_x)
    # system
    D = np.array([[0.0, 0.0, 0.0, 1.0]]).T  # constant
    sigV = np.diag([1e-12, 1e-3])

    def observe(self, x, u):
        y = np.array([[np.sin(x[0, 0]),
                       np.cos(x[0, 0]),
                       x[1, 0],
                       0.0]]).reshape((self.dim_y, 1))
        yu = self.D.dot(u)
        y += yu

        C = np.array([[np.cos(x[0, 0]),  0.0],
                      [-np.sin(x[0, 0]), 0.0],
                      [0.0,              1.0],
                      [0.0,              0.0]])
        c = y - C.dot(x) - yu
        return y, C, c, self.D

class PendulumLinearObservationDef(PendulumDef):

    y_key = ["$\\theta$", "$\dot{\\theta}$", "$u$"]
    y_whiten = [True, True]
    dim_xa = 2  # augmented for 'observation'

    x_noise = np.diag([1e-6, 1e-6, 0.0])
    y_noise = np.diag([1e-4, 1e-4])

    # states
    xg = np.array([[0.0, 0.0]]).T
    xag = xg
    # system
    sigV = np.diag([1e-12, 1e-3])

    C = np.vstack((np.eye(2), np.zeros((1, 2))))
    c = np.zeros((3,1))
    D = np.array([[0.0, 0.0, 1.0]]).T

    def observe(self, x, u):
        y = self.C.dot(x) + self.D.dot(u) + self.c
        return y, self.C, self.c, self.D

class PendulumKnown(PendulumDef):

    @staticmethod
    def dynamics(x, u):
        return dyn.pendulum_dynamics(x, u)

    @classmethod
    def dydx(cls, x, u):
        return dyn.pendulum_dydx(x, u).reshape((cls.dim_x, cls.dim_x))

    @classmethod
    def dydu(cls, x, u):
        return dyn.pendulum_dydu(x, u).reshape((cls.dim_x, cls.dim_u))

class PendulumLinearObservationKnown(PendulumLinearObservationDef):

    @staticmethod
    def dynamics(x, u):
        return dyn.pendulum_dynamics(x, u)

    @classmethod
    def dydx(cls, x, u):
        return dyn.pendulum_dydx(x, u).reshape((cls.dim_x, cls.dim_x))

    @classmethod
    def dydu(cls, x, u):
        return dyn.pendulum_dydu(x, u).reshape((cls.dim_x, cls.dim_u))


# class FurutaPendulumDef(object):

#     key = ["$\\theta$", "$\dot{\\theta}$", "$\\phi$", "$\dot{\\phi}$" "$u$"]
#     unit = ["rad", "rad/s", "rad", "rad/s", "Nm"]
#     dim_x = 4
#     dim_xa = 6  # augmented for 'observation'
#     dim_u = 1
#     dim_y = dim_xa + dim_u
#     # for model training
#     dim_xt = dim_x + dim_u
#     dim_yt = dim_x
#     # states
#     x0 = np.array([[np.pi, 0.0, 0.0, 0.0]]).T
#     xg = np.array([[0.0, 0.0, 0.0, 0.0]]).T
#     xag = np.array([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]).T
#     sg = np.vstack((xag, np.zeros((dim_u, dim_u))))
#     sgc = np.vstack((xg, np.zeros((dim_u, dim_u))))
#     sigX0 = 1e-6 * np.eye(dim_x)
#     # system
#     D = np.array([[0.0, 0.0, 0.0, 1.0]]).T  # constant


class BaseCartpoleDef(BaseDef):

    key = ["$x$", "$\\theta$", "$\dot{x}$", "$\dot{\\theta}$", "$u$"]
    y_key = ["$x$", "$\sin(\\theta)$", "$\cos(\\theta)$", "$\dot{x}$", "$\dot{\\theta}$", "$u$"]
    unit = ["m", "rad", "m/s", "rad/s", "Nm"]
    # environmental data scaling parameters
    x_whiten = [True, False, True, True, True]
    y_whiten = [True, True, True, True]
    # dimensions
    dim_x = 4
    dim_xa = 5  # augmented for 'observation'
    dim_u = 1
    # states
    sigX0 = 1e-6 * np.eye(dim_x)
    # system
    D = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T  # constant
    sigV = np.diag([1e-12, 1e-12, 5e-4, 5e-4])
    # data augmentation
    x_noise = np.diag([1e-9, 1e-9, 1e-9, 1e-9, 1e-9])
    y_noise = np.diag([1e-6, 1e-6, 1e-6, 1e-6])


    def observe(self, x, u):
        y = np.array([[x[0, 0],
                       np.sin(x[1, 0]),
                       np.cos(x[1, 0]),
                       x[2, 0],
                       x[3, 0],
                       0.0]]).reshape((self.dim_y, 1))
        yu = self.D.dot(u)
        y += yu

        C = np.array([[1.0,             0.0, 0.0, 0.0],
                      [0.0, np.cos(x[1, 0]), 0.0, 0.0],
                      [0.0,-np.sin(x[1, 0]), 0.0, 0.0],
                      [0.0,             0.0, 1.0, 0.0],
                      [0.0,             0.0, 0.0, 1.0],
                      [0.0,             0.0, 0.0, 0.0]])
        c = y - C.dot(x) - yu
        return y, C, c, self.D

class CartpoleDef(BaseCartpoleDef):

    x0 = np.array([[0.0, np.pi, 0.0, 0.0]]).T
    xg = np.array([[0.0, 0.0, 0.0, 0.0]]).T
    xag = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]).T
    sigV = np.diag([1e-12, 1e-12, 1e-6, 1e-6])

class CartpoleKnown(CartpoleDef):

    @staticmethod
    def dynamics(x, u):
        return dyn.cartpole_dynamics(x, u)

    @classmethod
    def dydx(cls, x, u):
        return dyn.cartpole_dydx(x, u).reshape((cls.dim_x, cls.dim_x))

    @classmethod
    def dydu(cls, x, u):
        return dyn.cartpole_dydu(x, u).reshape((cls.dim_x, cls.dim_u))


class QuanserCartpole(BaseCartpoleDef, dyn.QuanserCartpole):

    x0 = np.array([[0.0, 1e-5, 0.0, 0.0]]).T # QUANSER
    xg = np.array([[0.0, np.pi, 0.0, 0.0]]).T # QUANSER
    xag = np.array([[0.0, 0.0, -1.0, 0.0, 0.0]]).T # QUANSER
    # sigV = np.diag([1e-12, 1e-12, 1e-4, 1e-4])
    sigV = np.diag([1e-12, 1e-12, 1e-12, 1e-12])

    @classmethod
    def dynamics(cls, x, u):
        return cls.dynamics_fwd(x, u)

    @classmethod
    def dydx(cls, x, u):
        return cls.dynamics_dydx()(x, u).reshape((cls.dim_x, cls.dim_x))

    @classmethod
    def dydu(cls, x, u):
        return cls.dynamics_dydu()(x, u).reshape((cls.dim_x, cls.dim_u))

class DoubleCartpoleDef(BaseDef):

    key = ["$x$", "$\\theta_1$", "$\\theta_2$",
           "$\dot{x}$", "$\dot{\\theta}_1$", "$\dot{\\theta}_2$",
           "$u$"]
    y_key = ["$x$", "$\sin\\theta_1$", "$\cos\\theta_1$", "$\sin\\theta_2$", "$\cos\\theta_2$",
             "$\dot{x}$", "$\dot{\\theta}_1$", "$\dot{\\theta}_2$", "$u$"]
    unit = ["m", "rad", "rad", "m/s", "rad/s", "rad/s", "Nm"]
    x_whiten = [True, False, False, True, True, True, True]
    y_whiten = [True, False, False, True, True, True]
    dim_x = 6
    dim_xa = 8  # augmented for 'observation'
    dim_u = 1
    # states
    x0 = np.array([[0.0, np.pi, np.pi,
                    0.0, 0.0, 0.0]]).T
    xg = np.array([[0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0]]).T
    xag = np.array([[0.0, 0.0, 1.0, 0.0, 1.0,
                     0.0, 0.0, 0.0]]).T
    sg = np.vstack((xag, np.zeros((dim_u, dim_u))))
    sgc = np.vstack((xg, np.zeros((dim_u, dim_u))))
    sigX0 = 1e-6 * np.eye(dim_x)
    # system
    D = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T  # constant
    sigV = np.diag([1e-12, 1e-12, 1e-12, 1e-6, 1e-6, 1e-6])
    # sigV = np.diag([1e-12, 1e-12, 1e-12, 5e-4, 5e-4, 5e-4]) # visible
    # data augmentation
    x_noise = np.diag([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
    y_noise = np.diag([1e-8, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])

    def observe(self, x, u):
        y = np.array([[x[0, 0],
                       np.sin(x[1, 0]),
                       np.cos(x[1, 0]),
                       np.sin(x[2, 0]),
                       np.cos(x[2, 0]),
                       x[3, 0],
                       x[4, 0],
                       x[5, 0],
                       0.0]]).reshape((self.dim_y, 1))
        yu = self.D.dot(u)
        y += yu

        C = np.array([[1.0,             0.0,              0.0, 0.0, 0.0, 0.0],
                      [0.0, np.cos(x[1, 0]),              0.0, 0.0, 0.0, 0.0],
                      [0.0,-np.sin(x[1, 0]),              0.0, 0.0, 0.0, 0.0],
                      [0.0,             0.0,  np.cos(x[2, 0]), 0.0, 0.0, 0.0],
                      [0.0,             0.0, -np.sin(x[2, 0]), 0.0, 0.0, 0.0],
                      [0.0,             0.0,              0.0, 1.0, 0.0, 0.0],
                      [0.0,             0.0,              0.0, 0.0, 1.0, 0.0],
                      [0.0,             0.0,              0.0, 0.0, 0.0, 1.0],
                      [0.0,             0.0,              0.0, 0.0, 0.0, 0.0]])
        c = y - C.dot(x) - yu
        return y, C, c, self.D

class DoubleCartpoleKnown(DoubleCartpoleDef):

    @staticmethod
    def dynamics(x, u):
        return dyn.double_cartpole_dynamics(x.squeeze(), u.squeeze())

    @classmethod
    def dydx(cls, x, u):
        return dyn.double_cartpole_dydx(x.squeeze(), u.squeeze()).reshape((cls.dim_x, cls.dim_x))

    @classmethod
    def dydu(cls, x, u):
        return dyn.double_cartpole_dydu(x.squeeze(), u.squeeze()).reshape((cls.dim_x, cls.dim_u))


class TwoLinkElasticRobotDef(BaseDef):

    key = ["$\dot{q_1}$", "$\dot{q_2}$", "$\dot{\\theta_1}$", "$\dot{\\theta_2}$",
           "$q_1$", "$q_2$", "$\\theta_1$", "$\\theta_2$", "$u_1$", "$u_2$"]
    y_key = ["$\dot{q_1}$", "$\dot{q_2}$", "$\dot{\\theta_1}$", "$\dot{\\theta_2}$",
           "$p_x$", "$p_y$", "$d\\theta_1$", "$d\\theta_2$", "$u_1$", "$u_2$"]
    unit = ["rad/s", "rad/2", "rad/s", "rad/2",
            "rad", "rad", "rad", "rad", "Nm", "Nm"]
    # environmental data scaling parameters
    x_whiten = [True, False, True, True, True]
    y_whiten = [True, True, True, True]
    # dimensions
    dim_x = 8
    dim_xa = 8  # augmented for 'observation'
    dim_u = 2
    # for model training
    x_noise = 1e-5 * np.eye(10)
    y_noise = 1e-5 * np.eye(12)
    # states
    # like pendulum swing-up
    x0 = np.array([[0.0, 0.0,
                    0.0, 0.0,
                    np.pi, 0.0,
                    np.pi, 0.0]]).T
    xg = np.array([[0.0, 0.0,
                    0.0, 0.0,
                    0.0, 0.0,
                    0.0, 0.0]]).T
    xag = np.array([[0.0, 0.0,           # qd qd
                     0.0, 0.0,           # qd - thd
                     1.0, 0.0,           # end effector position
                     0.0, 0.0]]).T       # q - th
    sg = np.vstack((xag, np.zeros((dim_u, 1))))
    sgc = np.vstack((xg, np.zeros((dim_u, 1))))
    sigX0 = 1e-6 * np.eye(dim_x)
    # system
    D = np.vstack((np.zeros((dim_xa, dim_u)),
                   np.eye(dim_u)))   # constant


class TwoLinkElasticJointRobotKnown(TwoLinkElasticRobotDef):

    @staticmethod
    def dynamics(x, u):
        return dyn.two_link_elastic_joint_robot_arm_dynamics(x, u)#.reshape((cls.dim_x, 1))

    @classmethod
    def dydx(cls, x, u):
        # return dyn.two_link_dydx(x, u)#.reshape((cls.dim_x, cls.dim_x))
        J = dyn.two_link_dydx(x, u)
        print(J.shape)
        return J

    @classmethod
    def dydu(cls, x, u):
        return dyn.two_link_dydu(x, u).reshape((cls.dim_x, cls.dim_u))
