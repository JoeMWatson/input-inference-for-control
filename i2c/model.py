"""
Dynamic models for i2c (learnt and known).
A given model class with combined with an environment definition.
This allows us to do optimal control and model-based RL with the same API.
"""

import matplotlib.pyplot as plt
import os


import numpy as np
from numpy.random import multivariate_normal as mvn

from dataclasses import asdict

import i2c.env_def as env_def


def make_env_model(env_def, model_def):
    """
    :param env_def environment definition object from env_def.py
    :param model_def model definition (e.g. known, Gaussian process)
    :return appropraite defined model
    """
    _env_lookup = {
        "LinearKnown": LinearExact,
        "LinearKnownMinimumEnergy": LinearMinimumEnergy,
        "PendulumKnown": PendulumKnown,
        "PendulumKnownActReg": PendulumKnownActReg,
        "PendulumKnownLearn": PendulumLearn,
        "Cartpole_dmcs": CartpoleLearn,
        "CartpoleKnown": CartpoleKnown,
        "CartpoleKnownLearn": CartpoleLearn,
        "DoubleCartpoleKnown": DoubleCartpoleKnown,
        "DoubleCartpoleKnownLearn": DoubleCartpoleLearn,
    }
    _model_lookup = {
        type(None).__name__: None,  # known model
        # no learned models yet
    }
    _env = _env_lookup[env_def]
    _model = _model_lookup[type(model_def).__name__]
    model = _env(_model, model_def)
    return model


class BaseModel(object):
    """Base model object for known or learned models. Mixed with env definitions."""

    model = None
    data_driven = False

    def get(self):
        pass

    def load(self, parameters):
        pass

    def init(self):
        return self.x0.squeeze(), self.sig_x0

    def __call__(self, xu):
        return self.forward(xu)

    def run(self, horizon, policy, deterministic=False):
        XU = np.zeros((horizon, self.dim_xu))
        Z = np.zeros((horizon, self.dim_z))
        x = np.copy(self.x0).reshape((1, self.dim_x))
        for t in range(horizon):
            u = policy(t, x.T, deterministic=deterministic).T
            xu = np.hstack((x, u))
            XU[t, :] = xu
            Z[t, :] = self.observe(xu)[0, :]
            x = self.sample(xu)

        ZT = self.observe_terminal(x).squeeze()
        return XU, Z, ZT

    def predict(self, xu):
        raise NotImplementedError

    def forward(self, xu):
        # MuJoCo asserts at instability so clipping is necessary for forward rollouts
        # xu = np.clip(xu, self.xu_lim[0, :], self.xu_lim[1, :])
        _x = self.dynamics(xu)
        return _x, np.repeat(self.sig_eta[None, :, :], xu.shape[0], axis=0)

    def forward_linearize(self, xu):
        raise NotImplementedError

    def sample(self, xu):
        mu, sigma = self.forward(xu)
        eps = np.random.randn(xu.shape[0], self.dim_x)
        sigma_sqrt = np.linalg.cholesky(sigma)
        return mu + np.einsum("bij,bj->bi", sigma_sqrt, eps)

    def observe(self, xu):
        raise NotImplementedError

    def observe_terminal_x(self, x):
        return self.observe_terminal(x)

    def train(self, x, y, xt, yt, res_dir, name):
        if self.model is not None:
            self.model.train(x, y, xt, yt, res_dir, name)
        else:
            print("Known model, no training")

    def calibrate_epistemic(self, y):
        assert self.model is None, self.model

    def save(self, path):
        path = os.path.join(path, "sys.model")
        if self.model is not None:
            self.model.save(path)
        else:
            print("Known model, no saving")

    def train(self, X, Y):
        raise NotImplementedError

    def plot_sim(self, xu, xu_data=None, name="", res_dir=None):
        f, a = plt.subplots(self.dim_xu)
        for i, _a in enumerate(a):
            if xu_data is not None:
                for xu in xu_data:
                    _a.plot(xu[:, i], "c-", alpha=0.5)
            _a.plot(xu[:, i], "m-")

            _a.set_ylabel(
                f"{self.key[i]} ({self.unit[i]})" if self.unit[i] else f"{self.key[i]}"
            )
        a[0].set_title(f"Model simulation: {name}")
        a[-1].set_xlabel("Timesteps")
        if res_dir:
            plt.savefig(
                os.path.join(res_dir, f"model_sim_{name.replace(' ', '_')}.png"),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)


class BaseModelKnown(BaseModel):
    """Base model for known models, for optimal control."""

    data_driven = False

    def __init__(self, model=None, model_def=None):
        assert model is None
        assert model_def is None
        super().__init__()

    def forward(self, xu):
        _x = self.dynamics(xu)
        return _x, np.repeat(self.sig_eta[None, :, :], xu.shape[0], axis=0)

    def forward_linearize(self, xu):
        assert xu.shape == (1, self.dim_xu)
        _x = self.dynamics(xu).T
        AB = self.dydxu(xu)
        a = _x - AB @ xu.T
        A, B = AB[:, : self.dim_x], AB[:, self.dim_x :]
        return _x, A, B, a, self.sig_eta

    def predict(self, xu):
        return self.dynamics(xu)

    def predict_1d(self, x, u):
        # used for baselines
        xu = np.hstack((x, u)).reshape(
            (
                1,
                self.dim_xu,
            )
        )
        _x = self.dynamics(xu)
        # return np.clip(_x, self.xu_lim[0, :self.dim_x], self.xu_lim[1, :self.dim_x])
        return _x

    def train(self, X, Y):
        pass


class BaseModelLearn(BaseModel):
    """Base model for learned (i.e Bayesian) dynamics."""

    data_driven = True
    trained = False

    def __init__(self, model, model_def):
        super().__init__()
        self.model = model(self.dim_xt, self.dim_yt, **asdict(model_def))

    def get(self):
        return self.model.get()

    def load(self, parameters):
        self.model.load(parameters)

    def process(self, x):
        _x = np.clip(x, self.xu_lim[0, :], self.xu_lim[1, :])
        return self.featurespace(_x)

    def forward(self, xu):
        mean, covar = self.model.predict(self.process(xu))
        return mean + xu[:, : self.dim_x], covar

    def forward_linearize(self, x, u):
        raise NotImplementedError

    def predict(self, xu):
        return self.model.predict(self.process(xu))[0] + xu[:, : self.dim_x]

    def _predict(self, xu):
        m, c = self.model.predict(self.process(xu))
        return m, c

    def train(self, X, Y, name, res_dir):
        raise NotImplementedError

    def evaluate(self, X, Y):
        return None


class LinearBase(BaseModel):
    def __init__(self, model, model_def):
        assert model is None

    def dynamics(self, xu):
        return xu @ self.AB.T + self.a.T

    def predict(self, xu):
        return self.dynamics(xu)

    def forward(self, xu):
        _x = self.dynamics(xu)
        return _x, np.repeat(self.sig_eta[None, :, :], xu.shape[0], axis=0)

    def forward_linearize(self, xu):
        _x = self.dynamics(xu).T
        return _x, self.A, self.B, self.a, self.sig_eta


class LinearExact(env_def.LinearDef, LinearBase):
    """Linear dynamical system."""


class LinearMinimumEnergy(env_def.LinearMinimumEnergyDef, LinearBase):
    """Linear dynamical system with only control regularization."""


class PendulumKnown(env_def.PendulumKnown, BaseModelKnown):
    """NumPy pendulum swing-up."""


class PendulumKnownActReg(env_def.PendulumKnownActReg, BaseModelKnown):
    """NumPy pendulum swing-up with control regularization."""


class PendulumLearn(env_def.PendulumDef, BaseModelLearn):
    """NumPy pendulum with model learning."""


class FurutaLearn(env_def.FurutaDef, BaseModelLearn):
    """NumPy Furuta pendulum swing-up with model learning."""


class FurutaKnown(env_def.FurutaKnown, BaseModelKnown):
    """NumPy Furuta pendulum swing-up."""


class CartpoleLearn(env_def.CartpoleDef, BaseModelLearn):
    """NumPy Cartpole swing-up with model learning."""


class CartpoleKnown(env_def.CartpoleKnown, BaseModelKnown):
    """NumPy Cartpole swing-up."""


class DoubleCartpoleLearn(env_def.DoubleCartpoleDef, BaseModelLearn):
    """NumPy Double Cartpole swing-up with model learning."""


class DoubleCartpoleKnown(env_def.DoubleCartpoleKnown, BaseModelKnown):
    """NumPy Double Cartpole swing-up."""
