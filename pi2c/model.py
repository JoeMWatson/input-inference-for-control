"""
Dynamic models for i2c (learnt and known)
"""
import numpy as np

import pi2c.env_def as env_def
import pi2c.exp_types as types

def make_env_model(env_def, model_def):
    _env_lookup = {
        "LinearKnown": LinearExact,
        "PendulumKnown": PendulumKnown,
        "PendulumLinearObservationKnown": PendulumLinearObservationKnown,
        "CartpoleKnown": CartpoleKnown,
        "QuanserCartpoleKnown": QuanserCartpoleKnown,
        "DoubleCartpoleKnown": DoubleCartpoleKnown,
        "ElasticTwoLinkKnown": TwoLinkElasticJointRobotKnown
    }
    _model_lookup = {
        type(None).__name__: None, # known model
    }
    _env = _env_lookup[env_def]
    _model = _model_lookup[type(model_def).__name__]
    return _env(_model, model_def)


class BaseModel(object):

    model = None

    def init(self):
        return self.x0.squeeze(), self.sigX0

    def __call__(self, x, u):
        return self.forward(x, u)

    def forward(self, x, u):
        raise NotImplementedError

    def observe(self, x, u):
        raise NotImplementedError

    def train(self, x, y, xt, yt, res_dir, name):
        if self.model is not None:
            self.model.train(x, y, xt, yt, res_dir, name)
        else:
            print("Known model, no training")

    def load(self, path):
        if self.model is not None:
            self.model.load(path)
        else:
            print("Known model, no loading")

    def save(self, path):
        if self.model is not None:
            self.model.save(path)
        else:
            print("Known model, no saving")

class LinearExact(env_def.LinearDef, BaseModel):

    def __init__(self, model, model_def):
        assert model is None

    def dynamics(self, x, u):
        return self.A @ x + self.B @ u + self.a

    def forward(self, x, u):
        _x = self.dynamics(x, u)
        return _x, self.A, self.a, self.B, self.sigV

class PendulumKnown(env_def.PendulumKnown, BaseModel):

    def __init__(self, model, model_def):
        assert model is None
        assert model_def is None

    def forward(self, x, u):
        _x = self.dynamics(x, u)
        A = self.dydx(x, u)
        B = self.dydu(x, u)
        a = _x - A.dot(x) - B.dot(u)
        return _x, A, a, B, self.sigV

class PendulumLinearObservationKnown(env_def.PendulumLinearObservationKnown, BaseModel):

    def __init__(self, model, model_def):
        assert model is None
        assert model_def is None

    def forward(self, x, u):
        _x = self.dynamics(x, u)
        A = self.dydx(x, u)
        B = self.dydu(x, u)
        a = _x - A.dot(x) - B.dot(u)
        return _x, A, a, B, self.sigV

class CartpoleKnown(env_def.CartpoleKnown, BaseModel):

    def __init__(self, model, model_def):
        assert model is None
        assert model_def is None

    def __call__(self, x, u):
        return self.forward(x, u)

    def forward(self, x, u):
        _x = self.dynamics(x, u)
        A = self.dydx(x, u)
        B = self.dydu(x, u)
        a = _x - A.dot(x) - B.dot(u)
        return _x, A, a, B, self.sigV

class QuanserCartpoleKnown(env_def.QuanserCartpole, BaseModel):

    def __init__(self, model, model_def):
        assert model is None
        assert model_def is None

    def forward(self, x, u):
        _x = self.dynamics(x, u).reshape((self.dim_x, 1))
        A = self.dydx(x, u)
        B = self.dydu(x, u)
        a = _x - A.dot(x) - B.dot(u)
        return _x, A, a, B, self.sigV

class DoubleCartpoleKnown(env_def.DoubleCartpoleKnown, BaseModel):

    def __init__(self, model, model_def):
        assert model is None
        assert model_def is None

    def __call__(self, x, u):
        return self.forward(x, u)

    def forward(self, x, u):
        _x = self.dynamics(x, u).reshape((-1, 1))
        A = self.dydx(x, u)
        B = self.dydu(x, u)
        a = _x - A.dot(x) - B.dot(u)
        return _x, A, a, B, self.sigV

class TwoLinkElasticJointRobotKnown(BaseModel, env_def.TwoLinkElasticJointRobotKnown):

    def __init__(self, model, model_def):
        assert model is None
        assert model_def is None

    def __call__(self, x, u):
        return self.forward(x, u)

    def forward(self, x, u):
        _u = u.reshape((-1, 1))
        _x = self.dynamics(x, _u).squeeze()
        A = self.dydx(x, _u)
        B = self.dydu(x, _u)
        a = _x - A.dot(x) - B.dot(_u)
        sigV = 1e-12 * np.eye(self.dim_x)
        return _x, A, a, B, sigV

    def observe(self, x, u):
        y = np.array([[
                       # velocties
                       x[0, 0],
                       x[1, 0],
                       x[2, 0],
                       x[3, 0],
                       # angles
                       0.5 * np.cos(x[4, 0]) + 0.5 * np.cos(x[4, 0] + x[5, 0]),
                       0.5 * np.sin(x[4, 0]) + 0.5 * np.sin(x[4, 0] + x[5, 0]),
                       x[4, 0] - x[6, 0],
                       x[5, 0] - x[7, 0],
                       # inputs
                       0.0,
                       0.0]]).reshape((self.dim_y, 1))
        yu = self.D.dot(u)
        print(y.shape, yu.shape, self.D.shape, u.shape)
        y += yu

        dy4dx4 = -0.5 * np.sin(x[4, 0]) - 0.5 * np.sin(x[4, 0]+ x[5, 0])
        dy4dx5 = -0.5 * np.sin(x[4, 0] + x[5, 0])
        dy5dx4 = 0.5 * np.cos(x[4, 0]) + 0.5 * np.cos(x[4, 0]+ x[5, 0])
        dy5dx5 = 0.5 * np.cos(x[4, 0]+ x[5, 0])

        C = np.array([# Velocities
                      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      # Angles
                      [0.0, 0.0, 0.0, 0.0, dy4dx4, dy4dx5, 0.0,  0.0],
                      [0.0, 0.0, 0.0, 0.0, dy5dx4, dy5dx5, 0.0,  0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0,    0.0,    -1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0,    1.0,    0.0,  -1.0],
                      # Inputs
                      [0.0, 0.0, 0.0, 0.0, 0.0,    0.0,    0.0,  0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0,    0.0,    0.0,  0.0]])
        c = y - C.dot(x) - yu

        print(y.shape, C.shape, c.shape, yu.shape)
        return y, C, c
