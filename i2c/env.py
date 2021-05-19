"""
Environments to control.
Used environment definitions with a simulator, so its compatible with the models.
"""
import os
import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal as mvn
import multiprocessing as mp
from tqdm import tqdm
from scipy import signal

import i2c.env_def as env_def


def make_env(exp):
    """Returns the environemnt simulator from the experiment definition."""
    _lookup = {
        "LinearKnown": LinearSim,
        "LinearKnownMinimumEnergy": LinearMinimumEnergy,
        "PendulumKnown": PendulumKnown,
        "PendulumKnownActReg": PendulumKnownActReg,
        "PendulumKnownLearn": PendulumKnown,
        "CartpoleKnown": CartpoleKnown,
        "CartpoleSetKnown": CartpoleKnown,
        "CartpoleKnownLearn": CartpoleKnown,
        "DoubleCartpoleKnown": DoubleCartpoleKnown,
        "DoubleCartpoleKnownLearn": DoubleCartpoleKnown,
    }
    return _lookup[exp.ENVIRONMENT](exp.N_DURATION)


class BaseSim(object):
    """Base object for simulators"""

    env = None
    simulated = True

    def run(self, policy, deterministic=True, render=False, use_tqdm=False):
        """Run the environment with a given policy and gather data.

        :policy controller object
        :render True if render frames
        :use_tqdm True to disply progress bar
        :return training data (x, dx, z, z_term)
        :return training data (x, dx, z, z_term) and rendered images
        """
        frames = []
        if render:
            assert hasattr(self, "render")
        xt = np.zeros((self.duration, self.dim_s))
        yt = np.zeros((self.duration, self.dim_x))
        zt = np.zeros((self.duration, self.dim_z))
        x = self.init_env()
        iterator = tqdm(range(self.duration)) if use_tqdm else range(self.duration)
        for t in iterator:
            u = policy(t, x.T, deterministic).reshape((1, self.dim_u))
            x_prev = np.copy(x)
            if render:
                frames.append(np.array(self.render()))
            x = self.forward(u)
            xt_entry = np.hstack((x_prev, u))
            xt[t, :] = xt_entry[0, :]
            yt[t, :] = (x - x_prev)[0, :]
            zt[t, :] = self.observe(xt_entry)[0, :]
        # terminal observation for terminal cost
        # assumes that zero is goal state
        z_term = self.observe_terminal(x)
        yt = self.process_y(yt)  # needs filtering on real systems
        if render:
            return xt, yt, zt, z_term, frames
        else:
            return xt, yt, zt, z_term

    def run_render(self, policy, dir, name="", deterministic=True, use_tqdm=False):
        if hasattr(self, "render"):
            xt, yt, zt, z_term, stream = self.run(
                policy, deterministic=deterministic, render=True, use_tqdm=use_tqdm
            )
            gif_name = os.path.join(dir, f"render_{name}.gif")
            imageio.mimsave(gif_name, stream, fps=int(1 / self.timestep))
            optimize(gif_name)
            return xt, yt, zt, z_term
        else:
            print("Cannot render this environment")
            return self.run(policy, deterministic, render=False)

    def batch_run(self, args):
        policy, deterministic = args
        return self.run(policy, deterministic)

    def batch_eval(self, policy, n_eval, deterministic=True):
        args = [(policy, deterministic) for _ in range(n_eval)]
        if n_eval > 1:
            with mp.Pool(10) as p:  # TODO avoid this!!
                out = p.map(self.batch_run, args)
                # self.pool.map(self.batch_run, args)
                assert len(out) == n_eval
                return zip(*out)
        else:
            x, y, z, z_term = self.run(policy, deterministic)
            return [x], [y], [z], [z_term]

    def forward(self, u):
        raise NotImplementedError

    def observe(self, xu):
        raise NotImplementedError

    def init_env(self):
        raise NotImplementedError

    def plot_sim(self, x, x_est=None, name="", res_dir=None):
        batch = not isinstance(x, np.ndarray)  # tuple or list
        horizon = x[0].shape[0] if batch else x.shape[0]
        f, a = plt.subplots(self.dim_s)

        # plot limits
        for i, _a in enumerate(a):
            if np.isfinite(self.xu_lim[0, i]):
                _a.plot(self.xu_lim[0, i] * np.ones((horizon,)), "k")
            if np.isfinite(self.xu_lim[1, i]):
                _a.plot(self.xu_lim[1, i] * np.ones((horizon,)), "k")

        for i, _a in enumerate(a):
            if batch:
                for j, _x in enumerate(x):
                    _a.plot(
                        _x[:, i],
                        ".-",
                        alpha=0.4,
                        color="c" if x_est is not None else None,
                        label="True" if j == 0 else None,
                    )
            else:
                _a.plot(x[:, i], "c.-", label="True")
            if x_est is not None:
                _a.plot(x_est[:, i], "m.-", label="Planned")
            _a.legend()
            _a.set_ylabel(
                f"{self.key[i]} ({self.unit[i]})" if self.unit[i] else f"{self.key[i]}"
            )
        a[0].set_title(f"Simulation: {name}")
        a[-1].set_xlabel("Timesteps")
        if res_dir:
            plt.savefig(
                os.path.join(res_dir, f"sim_{name.replace(' ', '_')}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

        self.plot_trajectory(x, x_est, name, res_dir)

    def plot_trajectory(self, x, x_est=None, name="", res_dir=None):
        # custom trajectory plots
        pass

    def close(self):
        if self.env is not None:
            self.env.close()

    def process_y(self, y):
        return y


class BaseKnownSim(BaseSim):
    """Follows pattern of the Known autograd environments"""

    def __init__(self, duration):
        super().__init__()
        # TODO check for dynamics method and sig eta
        self.duration = duration

    def init_env(self):
        self.x = np.copy(self.x0.reshape((1, self.dim_x)))
        return self.x

    def forward(self, u):
        assert u.shape[0] == 1
        xu = np.hstack((self.x, u))
        self.x = self.dynamics(xu)
        if not self.deterministic:
            disturbance = mvn(np.zeros((self.dim_x,)), self.sig_eta, 1)
            self.x += disturbance
        return self.x

    def batch_forward(self, xu):
        x = self.dynamics(xu)
        if self.deterministic:
            return x
        else:
            noise = mvn(np.zeros((self.dim_x)), self.sig_eta, x.shape[0])
            return x + noise


class BaseLinear(BaseSim):
    def __init__(self, duration):
        self.duration = duration

    def init_env(self):
        self.x = mvn(self.x0.squeeze(), self.sig_x0, 1)
        return self.x

    def dynamics(self, xu):
        return xu @ self.AB.T + self.a.T

    def forward(self, u):
        xu = np.hstack((self.x, u))
        disturbance = mvn(np.zeros((self.dim_x)), self.sig_eta, 1)
        self.x = self.dynamics(xu) + disturbance
        return self.x


class LinearSim(env_def.LinearDef, BaseLinear):
    """Linear dynamical system."""


class LinearMinimumEnergy(env_def.LinearMinimumEnergyDef, BaseLinear):
    """Linear dynamical system with control regularization."""


class PendulumKnown(env_def.PendulumKnown, BaseKnownSim):
    """NumPy pendulum swing-up environment."""


class PendulumKnownActReg(env_def.PendulumKnownActReg, BaseKnownSim):
    """NumPy pendulum swing-up environment with control regularization."""


class CartpoleKnown(env_def.CartpoleKnown, BaseKnownSim):
    """NumPy cartpole swing-up simulator."""


class DoubleCartpoleKnown(env_def.DoubleCartpoleKnown, BaseKnownSim):
    """NumPy Double cartpole swing-up simulator."""
