"""
Self-contained experiment for MPC with state estimation for acrobatic quadrocopter
flight.
Uses box2d for historical (i.e. deadline motivated) reasons.
iLQR required some hacking to work in this setting.
"""

import os
import numpy as np

import Box2D
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)
from dataclasses import dataclass, asdict
import itertools
import gym
from gym import spaces


from i2c.env_def import BaseDef
from i2c.exp_types import CubatureQuadrature
from i2c.inference.quadrature import QuadratureInference
from i2c.model import BaseModelKnown
from baselines.ilqr import IterativeLqr


@dataclass
class Experiment(object):
    use_i2c: bool
    feedforward: bool
    low_noise: bool

    @property
    def algo(self):
        return "i2c" if self.use_i2c else "iLQR"

    @property
    def ctrl(self):
        return "FF" if self.feedforward else "FB"

    @property
    def noise(self):
        return "low" if self.low_noise else "high"

    @property
    def name(self):
        return f"{self.algo}_{self.ctrl}_{self.noise}"


# flag for visualization
RENDER = False
# RENDER = True

# the experiments look at i2c and ilqr MPC in low and high measurement noise settings
EXPERIMENTS = [
    Experiment(a, b, c) for a, b, c in itertools.product([True, False], repeat=3)
]


FS = 10

# render params
SCALE = 30.0
VIEWPORT_W = 600
VIEWPORT_H = 400
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

# quadrotor def
vehicle_dx = W / 25
vehicle_dy = H / 100
vehicle_poly = [
    (-vehicle_dx, -vehicle_dy),
    (-vehicle_dx, vehicle_dy),
    (vehicle_dx, vehicle_dy),
    (vehicle_dx, -vehicle_dy),
]


# need nd.Jacobian for box2d, autograd won't work
class AnalyticalLinearDynamics(object):
    def __init__(self, f_dyn, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps
        self.f = f_dyn
        self.A = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.B = np.zeros((self.dm_state, self.dm_act, self.nb_steps))

        self.dfdtau = nd.Jacobian(self.f)

    def evalf(self, x, u):
        return self.f(x, u)

    def taylor_expansion(self, x, u):
        for t in range(self.nb_steps):
            tau = np.hstack((x[:, t], u[:, t])).T
            AB = self.dfdtau(tau)
            self.A[..., t] = AB[:, : self.dm_state]
            self.B[..., t] = AB[:, self.dm_state :]


class IlqrMpc(object):
    """iLQR needed lots of hacking for Box2D, as its designed for autograd."""

    def __init__(self, ilqr, n_iter, z_traj=None):
        model = ilqr.env
        self.dim_u, self.dim_x = model.dim_u, model.dim_x
        self.model = model
        self.n_iter = n_iter
        self.z_traj = z_traj
        self.ilqr = ilqr
        if z_traj is not None:
            self.ilqr.weighting = z_traj[: self.ilqr.nb_steps + 1, :, None]
        self.xu_history = []
        self.z_history = []

        self.mu = model.x0
        self.covar = model.sig_x0
        self.mus = []
        self.covars = []
        inference = CubatureQuadrature(1, 0, 0)
        self.dyn_inf = QuadratureInference(inference, self.dim_x)
        self.meas_inf = QuadratureInference(inference, self.dim_x)

    def set_control(self, feedforward):
        self.ilqr.feedforward = feedforward

    def filter(self, y, u):
        assert u.shape == (self.dim_u, 1)
        # pass old belief through dynamics
        # expose quadrature to add control manually
        x_pts = self.dyn_inf.get_x_pts(self.mu, self.covar)
        x_pts = np.concatenate((x_pts, u.T.repeat(x_pts.shape[0], axis=0)), axis=1)
        x_f_pts, _sig_y = self.model.forward(x_pts)
        mu_f = np.einsum("b, bi->i", self.dyn_inf.weights_sig, x_f_pts).reshape((-1, 1))
        _sig_f = np.einsum(
            "b,bi,bj->ij", self.dyn_inf.weights_sig, x_f_pts, x_f_pts
        ) - np.outer(mu_f, mu_f)
        sig_eta = np.einsum("b,bij->ij", self.dyn_inf.weights_sig, _sig_y)
        sig_f = _sig_f + sig_eta

        # innovate on new measurement
        mu_y, sig_y = self.meas_inf.forward(self.model.measure, mu_f.T, sig_f)
        sig_y += self.model.sig_zeta
        K = np.linalg.solve(sig_y.T, self.meas_inf.sig_xy.T).T
        self.mu = mu_f + K @ (y - mu_y)
        self.covar = sig_f - K @ sig_y @ K.T
        return self.mu, self.covar

    def optimize(self, n_iter, mu):
        assert mu.shape == (self.dim_x, 1), f"{mu.shape}, {(self.dim_x, 1)}"
        self.ilqr.env_init = mu[:, 0]
        cost = self.ilqr.run(n_iter)
        print(self.ilqr.env_init[:2], cost)

    def __call__(self, i, y, u):
        if i > 0:
            self.filter(y, u)
        mu, covar = self.mu, self.covar
        self.mus.append(mu)
        self.covars.append(covar)
        self.optimize(self.n_iter, mu)
        self.xu_history.append(
            np.concatenate((self.ilqr.xref[:, :-1].T, self.ilqr.uref.T), axis=1)
        )
        mu = np.copy(self.ilqr.uref[:, 0])
        self.ilqr.ctl.kff[:, 0:-1] = self.ilqr.ctl.kff[:, 1:]
        self.ilqr.ctl.kff[:, -1] = self.ilqr.ctl.kff[:, -2]
        self.ilqr.uref[:, 0:-1] = self.ilqr.uref[:, 1:]
        self.ilqr.uref[:, -1] = self.ilqr.uref[:, -2]

        if self.z_traj is not None:
            self.ilqr.weighting[0:-1] = self.ilqr.weighting[1:]
            if (i + self.ilqr.nb_steps) < self.z_traj.shape[0]:
                self.ilqr.weighting[-1] = self.z_traj[i + self.ilqr.nb_steps, :, None]
            else:
                self.ilqr.weighting[-1] = self.ilqr.weighting[-2]

        return mu.reshape((self.dim_u, 1))

    def plot_history(self, res_path, name=""):
        horizon = len(self.xu_history)
        hist = np.asarray(self.xu_history)
        f, a = plt.subplots(self.dim_x + self.dim_u)
        for i, _a in enumerate(a):
            for t in range(horizon):
                _t = np.arange(t, t + self.ilqr.nb_steps)
                _a.plot(_t, hist[t, :, i])
        plt.savefig(
            join(res_path, f"xu_history_{name}.png"), bbox_inches="tight", format="png"
        )
        plt.close(f)


def add_tuple(t1, t2, sf):
    return tuple(map(lambda i, j: i + sf * j, t1, t2))


# unused in the end
class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass


class QuadrotorDef(BaseDef):
    """Environment definition for the quadcopter."""

    name = "2D Quadrator"
    key = [
        "$x$",
        "$y$",
        "$\psi$",
        "$\\dot{x}$",
        "$\\dot{y}$",
        "\\dot{\psi}",
        "u_1",
        "u_2",
    ]
    z_key = key
    unit = [None for _ in range(9)]

    deterministic = False
    dim_x = 6
    dim_u = 2
    dim_z = 8
    dim_y = 8
    dim_z_term = dim_z - dim_u

    x0 = np.array([[W / 4, H / 2, 0.0, 0.0, 0.0, 0.0]]).T
    sig_x0 = 1e-5 * np.eye(dim_x)
    sig_eta = np.diag([1e-6] * 2 + [1e-6] + [1e-4] * 2 + [1e-4])

    # measurement noise
    sig_zeta = None  # set in experiment

    xag = np.array([[3 * W / 4, H / 2, 0.0, 0.0, 0.0, 0.0]]).T
    zg_term = xag
    zgc = np.array([None] * dim_x + [0.0] * dim_u)

    force_mx = 30.0
    xu_lim = np.array(
        [
            [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, 0.0, 0.0],
            [np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, force_mx, force_mx],
        ]
    )

    def init_world(self):
        self.world = Box2D.b2World(gravity=(0, -9.81))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self.field = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []

        world_points = [(0, 0), (W, 0), (W, H), (0, H)]
        world_points.append(world_points[0])
        for p1, p2 in zip(world_points[:-1], world_points[1:]):
            self.field.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)

        x_0 = self.x0[0, 0]
        y_0 = self.x0[1, 0]
        self.x_g = self.xag[0, 0]
        self.y_g = self.xag[1, 0]
        self.lander = self.world.CreateDynamicBody(
            position=(x_0, y_0),
            angle=0.0,
            angularDamping=0.5,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=vehicle_poly),
                density=5.0,
                friction=0.0,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (1.0, 0.0, 1.0)
        self.lander.color2 = (0.0, 0.1, 0.0)
        self.lander.color3 = (0.0, 1.0, 1.0)
        self.lander.color4 = (0.7, 0.6, 0.5)

    def __init__(self, *args, **kwargs):
        self.init_world()
        super().__init__(*args, **kwargs)

    @property
    def left_pos(self):
        return (
            self.lander.position[0] - vehicle_dx * np.cos(self.lander.angle),
            self.lander.position[1] - vehicle_dx * np.sin(self.lander.angle),
        )

    @property
    def right_pos(self):
        return (
            self.lander.position[0] + vehicle_dx * np.cos(self.lander.angle),
            self.lander.position[1] + vehicle_dx * np.sin(self.lander.angle),
        )

    @property
    def thrust(self):
        return -np.sin(self.lander.angle), np.cos(self.lander.angle)

    @property
    def gravity(self):
        return 9.81 * self.lander.mass

    def step(self, x, u):
        u = self.clip_u(u)
        self.lander.position[0] = x[0]
        self.lander.position[1] = x[1]
        self.lander.angle = x[2]
        self.lander.linearVelocity[0] = x[3]
        self.lander.linearVelocity[1] = x[4]
        self.lander.angularVelocity = x[5]

        self.lander.ApplyForce(tuple(u[0] * np.array(self.thrust)), self.left_pos, True)
        self.lander.ApplyForce(
            tuple(u[1] * np.array(self.thrust)), self.right_pos, True
        )

        self.world.Step(1.0 / FS, 1, 1)

        return np.array(
            [
                self.lander.position[0],
                self.lander.position[1],
                self.lander.angle,
                self.lander.linearVelocity[0],
                self.lander.linearVelocity[1],
                self.lander.angularVelocity,
            ]
        )

    def dynamics(self, xu):
        # simulate dynamics with quadrature support
        # x is (n_pts, dim_x)
        n_s = xu.shape[0]
        _x = np.zeros((n_s, self.dim_x))

        for i in range(n_s):
            _x[i, :] = self.step(xu[i, : self.dim_x], xu[i, self.dim_x :])
        return _x

    @staticmethod
    def observe(xu):
        return xu

    @staticmethod
    def observe_terminal(x):
        return x

    @staticmethod
    def measure(x):
        """Observe position and velocity of left and right position."""
        lx = x[:, 0, None] - vehicle_dx * np.cos(x[:, 2, None])
        ly = x[:, 1, None] - vehicle_dx * np.sin(x[:, 2, None])
        lxd = x[:, 3, None] - vehicle_dx * -np.sin(x[:, 2, None]) * x[:, 5, None]
        lyd = x[:, 4, None] - vehicle_dx * np.cos(x[:, 2, None]) * x[:, 5, None]

        rx = x[:, 0, None] + vehicle_dx * np.cos(x[:, 2, None])
        ry = x[:, 1, None] + vehicle_dx * np.sin(x[:, 2, None])
        rxd = x[:, 3, None] + vehicle_dx - np.sin(x[:, 2, None]) * x[:, 5, None]
        ryd = x[:, 4, None] + vehicle_dx + np.cos(x[:, 2, None]) * x[:, 5, None]
        y = np.concatenate((lx, ly, rx, ry, lxd, lyd, rxd, ryd), axis=1)
        return y


class QuadrotorKnown(QuadrotorDef, BaseModelKnown):
    """Model from Quadrotor from def."""


class Quadrotor(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FS}

    continuous = False

    def __init__(self):
        self.viewer = None

        self.env = QuadrotorDef()
        self.env.init_world()
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([30.0, 30.0]), dtype=np.float32
        )

    def reset(self):
        self.state = self.env.x0[:, 0]
        return self.state[:, None], self.measure()

    def measure(self):
        noise = np.random.multivariate_normal(
            np.zeros((self.env.dim_y,)), self.env.sig_zeta, 1
        ).T
        return self.env.measure(self.state[None, :]).T + noise

    def step(self, action):
        self.action = action
        noise = np.random.multivariate_normal(
            np.zeros((self.env.dim_x,)), self.env.sig_eta, 1
        ).T
        self.state = self.env.step(self.state, action) + noise[:, 0]
        return self.state[:, None], self.measure()

    def render(self, mode="human", i2c=None, ilqr=None, z_traj=None):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, W, 0, H)

        if z_traj is not None:
            for z in z_traj:
                goal_transform = rendering.Transform(
                    rotation=0, translation=(z[0], z[1])
                )
                circ = self.viewer.draw_circle(0.05)
                # circ.set_color(1.0, 0, 0)
                circ._color.vec4 = (1.0, 0, 0, 0.1)
                circ.add_attr(goal_transform)

        self.lander = self.env.lander
        for obj in [self.lander]:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)
                path.append(path[0])
                self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        if ilqr is not None:
            for t in range(ilqr.weighting.shape[0]):
                _t = rendering.Transform(
                    rotation=0, translation=(ilqr.weighting[t, 0], ilqr.weighting[t, 1])
                )
                circ = self.viewer.draw_circle(0.1)
                circ.set_color(0, 0.5, 0.5)
                circ.add_attr(_t)

            for t in range(ilqr.xref.shape[1]):
                _t = rendering.Transform(
                    rotation=0, translation=(ilqr.xref[0, t], ilqr.xref[1, t])
                )
                circ = self.viewer.draw_circle(0.1)
                circ.set_color(0.5, 0.5, 0.5)
                circ.add_attr(_t)

        if i2c is not None:
            # state est
            _t = rendering.Transform(
                rotation=0, translation=(i2c.sys.x0[0, 0], i2c.sys.x0[1, 0])
            )
            circ = self.viewer.draw_circle(0.1)
            circ.set_color(0.5, 0.5, 0.5)
            circ.add_attr(_t)

            for c in i2c.cells[:-1]:
                # plot plan
                _t = rendering.Transform(
                    rotation=0, translation=(c.mu_x0_m[0, 0], c.mu_x0_m[1, 0])
                )
                circ = self.viewer.draw_circle(0.1)
                circ.set_color(0, 0, 0)
                circ.add_attr(_t)
                # plot quad
                pts = c.dyn_inf.y_pts
                for i in range(pts.shape[0]):
                    _t = rendering.Transform(
                        rotation=0, translation=(pts[i, 0], pts[i, 1])
                    )
                    circ = self.viewer.draw_circle(0.05)
                    circ.set_color(0, 1.0, 0)
                    circ.add_attr(_t)

        for p in self.env.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        self.viewer.draw_polyline(
            [
                self.env.left_pos,
                add_tuple(
                    self.env.left_pos,
                    self.env.thrust,
                    2 * self.action[0] / self.env.gravity,
                ),
            ],
            color=obj.color3,
            linewidth=2,
        )
        self.viewer.draw_polyline(
            [
                self.env.right_pos,
                add_tuple(
                    self.env.right_pos,
                    self.env.thrust,
                    2 * self.action[1] / self.env.gravity,
                ),
            ],
            color=obj.color3,
            linewidth=2,
        )

        return (
            self.viewer.render(return_rgb_array=mode == "rgb_array"),
            self.viewer.get_array(),
        )

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def run_experiment(seed):

    for exp in EXPERIMENTS:
        name = f"{exp.name}_{seed}"
        single_experiment(**asdict(exp), seed=seed, name=name)


def single_experiment(use_i2c, feedforward, low_noise, seed, name):
    np.random.seed(seed)

    res_dir = join(dirname(realpath(__file__)), "_results")

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    if exists(join(res_dir, f"{name}.npy")):  # have result
        print(f"{name} already done")
        return

    env = Quadrotor()
    model = QuadrotorKnown()
    sig_zeta = (
        np.diag([1e-6] * 8) if low_noise else np.diag([1e-6] * 2 + [5e-5] * 2 + [1] * 4)
    )
    env.env.sig_zeta = sig_zeta
    model.sig_zeta = sig_zeta

    T = 100
    T_plan = 10
    mpc_iter = 2

    # trajectory to follow -> sine with 360 pin
    z_traj = np.zeros((T, model.dim_z))
    z_traj[:, 0] = np.linspace(W / 4, 3 * W / 4, T)
    z_traj[:, 1] = H / 2 + (H / 4) * np.sin(np.linspace(0, 2 * np.pi, T))
    z_traj[:, 2] = 2 * np.pi * np.heaviside(np.linspace(-1, 1, T), 1)

    # tracking controller
    Q = np.diag([1e3, 1e3, 1e3, 1, 1, 1])
    R = np.diag([1e-3, 1e-3])
    QR = la.block_diag(Q, R) / 1e3
    Qf = Q / 1e3

    u_init = 0.5 * model.gravity * np.ones((T_plan, model.dim_u))
    if use_i2c:
        sig_u = 1e-2 * np.eye(model.dim_u)
        _i2c = I2cGraph(
            sys=model,
            horizon=T_plan,
            Q=Q,
            R=R,
            Qf=Qf,
            alpha=1.0,
            alpha_update_tol=1.0,
            mu_u=u_init,
            sig_u=sig_u,
            mu_x_terminal=None,
            sig_x_terminal=None,
            inference=CubatureQuadrature(1, 0, 0),
            res_dir=res_dir,
        )
        _i2c._propagate = True  # used for alpha calibration

        policy = PartiallyObservedMpcPolicy(_i2c, mpc_iter, sig_u, np.copy(z_traj))
    else:

        def cost(x, u, a):
            tau = np.hstack((x, u))
            a = a[:, 0]
            return (tau - a).T @ QR @ (tau - a)

        _ilqr = IterativeLqr(
            env=model,
            cost=cost,
            horizon=T_plan,
            u_lim=np.array([[0.0, 0.0], [30.0, 30.0]]),
        )
        # init with gravity comp.
        _ilqr.uref = u_init.T

        # nd.Jacobian only takes one argument in order to  work!!
        def dyn(tau):
            return model.step(tau[:6], tau[6:])

        _ilqr.dyn = AnalyticalLinearDynamics(
            dyn, _ilqr.dm_state, _ilqr.dm_act, _ilqr.nb_steps
        )

        policy = IlqrMpc(_ilqr, mpc_iter, np.copy(z_traj))

    policy.set_control(feedforward=feedforward)
    x, y = env.reset()

    warm_start_iter = 25  # 100
    if use_i2c:
        policy.i2c.calibrate_alpha()
        print(f"calibrated alpha: {policy.i2c.alpha:.2f}")
        policy.optimize(warm_start_iter, model.x0, model.sig_x0)
        policy.i2c.calibrate_alpha()
        print(f"recalibrated alpha: {policy.i2c.alpha:.2f}")
    else:
        print("ilqr warm start start")
        policy.ilqr.run(warm_start_iter)
        print("ilqr warm start done")
        policy.ilqr.dir_name = res_dir
        policy.ilqr.plot_trajectory("ilqr_warm_start")

    u = np.zeros((model.dim_u, 1))

    states = np.zeros((T, model.dim_s))
    obs = np.zeros((T, model.dim_y))
    stream = []
    for t in range(T):
        u = policy(t, y, u)
        u = model.clip_u(u.T).T
        states[t, :6] = x[:, 0]
        states[t, 6:] = u[:, 0]
        obs[t, :] = y[:, 0]
        x, y = env.step(np.asarray(u.flatten(), dtype=np.float))

        if RENDER:
            still_open, img = env.render(
                i2c=policy.i2c if use_i2c else None,
                ilqr=policy.ilqr if not use_i2c else None,
                z_traj=z_traj,
            )
            stream.append(img)

    err = states - z_traj
    cost = np.einsum("bi,ij,bi->", err, QR, err)
    print(cost)
    np.save(join(res_dir, f"{name}"), cost)
    np.save(join(res_dir, f"state_{name}"), states)
    np.save(join(res_dir, f"obs_{name}"), obs)

    if RENDER:
        gif_name = join(res_dir, f"{name}_render.gif")
        imageio.mimsave(gif_name, stream, fps=FS)
        optimize(gif_name)

    mus = np.asarray(policy.mus).reshape((T, model.dim_x))
    covars = np.asarray(policy.covars).reshape((T, model.dim_x, model.dim_x))

    f, ax = plt.subplots(model.dim_x, 2)
    for i, a in enumerate(ax[:, 0]):
        a.plot(states[:, i], "b-")
        a.plot(mus[:, i], "c--")

    for i, a in enumerate(ax[:, 1]):
        a.plot(np.sqrt(covars[:, i, i]), "c--")
    plt.savefig(
        join(res_dir, f"{name}_state_estimation.png"), bbox_inches="tight", format="png"
    )
    plt.close(f)

    f, ax = plt.subplots(1, 3)
    a = ax[0]
    a.plot(z_traj[:, 0], z_traj[:, 1], "m")
    a.plot(states[:, 0], states[:, 1], "b-")
    a.plot(mus[:, 0], mus[:, 1], "c--")
    for t in range(obs.shape[0]):
        a.plot(obs[t, [0, 2]], obs[t, [1, 3]], "y")
    a.set_ylim(0, H)
    a.set_xlim(0, W)
    a.set_ylabel("$y$")
    a.set_xlabel("$x$")

    a = ax[1]
    a.plot(z_traj[:, 2], "m")
    a.plot(states[:, 2], "b-")
    a.plot(mus[:, 2], "c--")
    a.set_xlabel("Timesteps")
    a.set_ylabel("$\psi$")

    a = ax[2]
    a.plot(states[:, 6], "c--", label="$u_1$")
    a.plot(states[:, 7], "b--", label="$u_2$")
    a.set_xlabel("Timesteps")
    a.set_ylabel("$u$")

    plt.savefig(
        join(res_dir, f"{name}_mpc_summary.png"), bbox_inches="tight", format="png"
    )
    plt.close(f)

    # use for debug
    # if use_i2c:
    #     policy.optimize(1, policy.mu, policy.covar)
    #     policy.i2c.plot_metrics(0, T, res_dir, "diag")
    #
    # policy.plot_history(res_dir, "i2c" if use_i2c else "ilqr")


if __name__ == "__main__":
    import argparse
    from os.path import dirname, realpath, join, exists
    import matplotlib.pyplot as plt
    from i2c.policy.mpc import PartiallyObservedMpcPolicy
    from i2c.i2c import I2cGraph
    import scipy.linalg as la
    import imageio
    from pygifsicle import optimize
    import numdifftools as nd

    import matplotlib

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["figure.figsize"] = [10, 10]
    matplotlib.rcParams["legend.fontsize"] = 16
    matplotlib.rcParams["axes.titlesize"] = 22
    matplotlib.rcParams["axes.labelsize"] = 22

    parser = argparse.ArgumentParser(description="Run Experiment")
    parser.add_argument("seed", help="random seed", default=0, type=int)
    parser.add_argument(
        "--plot", help="show any live plots at the end", action="store_true"
    )
    args = parser.parse_args()
    if args.plot:
        RENDER = True

    run_experiment(args.seed)

    if args.plot:
        plt.show()
