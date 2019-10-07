"""
Run baseline algorithms (GPS and iLQR)
"""
import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse
from autograd import jacobian
import autograd.numpy as np
import dill
from shutil import copyfile

DIR_NAME = os.path.dirname(__file__)
try:
    import pi2c
except ImportError:
    print("pi2c not installed, using local version")
    top_path = os.path.join(DIR_NAME, '..')
    sys.path.append(os.path.abspath(top_path))

from pi2c.env import make_env
from pi2c.model import make_env_model
from pi2c.policy.linear import TimeIndexedLinearGaussianPolicy
from scripts.experiments import pendulum_known, cartpole_known, double_cartpole_known

from baselines.gps import GuidedPolicySearch
from baselines.ilqr import IterativeLqr


def save_model(model, name, save_dir):
        filename = "%s.pkl" % name
        with open(os.path.join(save_dir, filename), 'wb') as f:
            dill.dump(model, f)


def pendulum_obs(x):
    return np.hstack((np.sin(x[0]), np.cos(x[0]), x[1]))

pendulum_dobs = jacobian(pendulum_obs)


def cartpole_obs(x):
    return np.hstack((x[0], np.sin(x[1]), np.cos(x[1]), x[2], x[3]))

cartpole_dobs = jacobian(cartpole_obs)

def double_cartpole_obs(x):
    return np.hstack((x[0],
                      np.sin(x[1]), np.cos(x[1]),
                      np.sin(x[2]), np.cos(x[2]),
                      x[3], x[4], x[5]))


double_cartpole_dobs = jacobian(double_cartpole_obs)

cartpole_dobs = jacobian(cartpole_obs)

def ilqr_pendulum(res_dir):
    experiment = pendulum_known
    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT,
                           experiment.MODEL)
    policy = TimeIndexedLinearGaussianPolicy(
        0 * np.eye(model.dim_u),
        experiment.N_DURATION, model.dim_u, model.dim_x)
    alpha = 1e4
    Q = experiment.INFERENCE.Q / alpha
    R = experiment.INFERENCE.R / alpha
    xg = model.xag.squeeze()

    def cost(x, u, a, x_lin):
        _J = pendulum_dobs(x_lin)
        _j = pendulum_obs(x_lin) - _J @ x_lin
        _x = _J @ x + _j
        return (_x - xg).T @ Q @ (_x - xg) + u.T @ R @ u

    ilqr = IterativeLqr(
        model, cost, experiment.N_DURATION, 2,
        alphas=np.power(10., np.linspace(0, -10, 50)),
        # mult_lmbda=1.6, # original matlab heuristic
        # mult_lmbda=1.001, good but needs 120 to hit 1.7
        mult_lmbda=1.002, # gets to 1.7 by 80
        tolgrad=1e-14,
        tolfun=1e-14,
        activation='all')
    # cost, cost_iter, full_cost = ilqr.run(200)
    cost, cost_real = ilqr.run(
        100
        )

    x, u, c = ilqr.forward_pass(ilqr.ctl, 0.0)
    print(x.shape, u.shape)
    xu = np.vstack((x[:,:-1], u)).T
    env.plot_sim(xu, None, "ilqr pendulum internal", res_dir)

    cost = [c * alpha for c in cost]
    cost_real = [c * alpha for c in cost_real]

    policy.K = np.zeros(policy.K.shape)
    policy.k = ilqr.uref.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "ilqr pendulum feedforward", None)


    policy.K = ilqr.ctl.K.reshape(policy.K.shape)
    # policy.k = ilqr.ctl.kff.reshape(policy.k.shape)
    policy.k = np.copy(ilqr.uref.reshape(policy.k.shape))
    Kx = np.einsum("jki,ki->ji", ilqr.ctl.K, ilqr.xref[:,:-1])
    policy.k += -Kx.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "ilqr pendulum", None)

    x = xu[:, :model.dim_x]
    u = xu[:, model.dim_x:]

    ilqr.plot_trajectory()

    f,a = plt.subplots(2)
    a[0].plot(cost, '.-')
    a[1].plot(cost_real, '.-')
    plt.savefig(os.path.join(res_dir, "cost_ilqr_pendulum.png"),
                bbox_inches='tight', format='png')

    x = ilqr.xref[:,:-1].T
    u = ilqr.uref.T

    return x, u, cost, ilqr

def ilqr_cartpole(res_dir):
    experiment = cartpole_known
    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT,
                           experiment.MODEL)
    policy = TimeIndexedLinearGaussianPolicy(
        0 * np.eye(model.dim_u),
        experiment.N_DURATION, model.dim_u, model.dim_x)
    alpha = 1e3
    Q = experiment.INFERENCE.Q / alpha
    R = experiment.INFERENCE.R / alpha
    xg = model.xag.squeeze()

    def cost(x, u, a, x_lin):
        _J = cartpole_dobs(x_lin)
        _j = cartpole_obs(x_lin) - _J @ x_lin
        _x = _J @ x + _j
        return (_x - xg).T @ Q @ (_x - xg) + u.T @ R @ u

    ilqr = IterativeLqr(
        model, cost, experiment.N_DURATION, 5,
        mult_lmbda=1.001,
        init_noise=1e-2,
        alphas=np.power(10., np.linspace(0, -8, 50)),
        tolgrad=1e-16,
        tolfun=1e-16,
        activation='all')
    cost, cost_real = ilqr.run(
        200
        )

    x, u, c = ilqr.forward_pass(ilqr.ctl, 0.0)
    print(x.shape, u.shape)
    xu = np.vstack((x[:,:-1], u)).T
    env.plot_sim(xu, None, "ilqr pendulum internal", res_dir)

    cost = [c * alpha for c in cost]
    cost_real = [c * alpha for c in cost_real]

    policy.K = np.zeros(policy.K.shape)
    policy.k = ilqr.uref.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "ilqr pendulum feedforward", None)


    policy.K = ilqr.ctl.K.reshape(policy.K.shape)
    policy.k = np.copy(ilqr.uref.reshape(policy.k.shape))
    Kx = np.einsum("jki,ki->ji", ilqr.ctl.K, ilqr.xref[:,:-1])
    policy.k += -Kx.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "ilqr pendulum", None)

    x = xu[:, :model.dim_x]
    u = xu[:, model.dim_x:]

    ilqr.plot_trajectory()

    f,a = plt.subplots(2)
    a[0].plot(cost, '.-')
    a[1].plot(cost_real, '.-')
    plt.savefig(os.path.join(res_dir, "cost_ilqr_cartpole.png"),
                bbox_inches='tight', format='png')

    x = ilqr.xref[:,:-1].T
    u = ilqr.uref.T

    return x, u, cost, ilqr

def ilqr_double_cartpole(res_dir):
    experiment = double_cartpole_known
    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT,
                           experiment.MODEL)
    policy = TimeIndexedLinearGaussianPolicy(
        0 * np.eye(model.dim_u),
        experiment.N_DURATION, model.dim_u, model.dim_x)
    alpha = 1e3
    Q = experiment.INFERENCE.Q / alpha
    R = experiment.INFERENCE.R / alpha
    xg = model.xag.squeeze()

    def cost(x, u, a, x_lin):
        _J = double_cartpole_dobs(x_lin)
        _j = double_cartpole_obs(x_lin) - _J @ x_lin
        _x = _J @ x + _j
        return (_x - xg).T @ Q @ (_x - xg) + u.T @ R @ u

    ilqr = IterativeLqr(
        model, cost, experiment.N_DURATION, 1e9,
        mult_lmbda=1.001,
        init_noise=1e-2,
        alphas=np.power(10., np.linspace(0, -8, 50)),
        tolgrad=1e-7,
        tolfun=1e-7,
        activation='all')
    cost, cost_real = ilqr.run(
        200
        )

    x, u, c = ilqr.forward_pass(ilqr.ctl, 0.0)
    xu = np.vstack((x[:,:-1], u)).T
    env.plot_sim(xu, None, "ilqr double cartpole internal", res_dir)

    cost = [c * alpha for c in cost]
    cost_real = [c * alpha for c in cost_real]

    policy.K = np.zeros(policy.K.shape)
    policy.k = ilqr.uref.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "ilqr double cartpole feedforward", res_dir)


    policy.K = ilqr.ctl.K.reshape(policy.K.shape)

    policy.k = np.copy(ilqr.uref.reshape(policy.k.shape))
    Kx = np.einsum("jki,ki->ji", ilqr.ctl.K, ilqr.xref[:,:-1])
    policy.k += -Kx.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "ilqr double cartpole", res_dir)

    x = xu[:, :model.dim_x]
    u = xu[:, model.dim_x:]

    ilqr.plot_trajectory()

    f,a = plt.subplots(2)
    a[0].plot(cost, '.-')
    a[1].plot(cost_real, '.-')
    plt.savefig(os.path.join(res_dir, "cost_ilqr_double_cartpole.png"),
                bbox_inches='tight', format='png')

    x = ilqr.xref[:,:-1].T
    u = ilqr.uref.T

    return x, u, cost, ilqr

def gps_pendulum(res_dir):
    experiment = pendulum_known
    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT,
                           experiment.MODEL)
    policy = TimeIndexedLinearGaussianPolicy(
        0 * np.eye(model.dim_u),
        experiment.N_DURATION, model.dim_u, model.dim_x)
    alpha = 1e4
    Q = experiment.INFERENCE.Q / alpha
    R = experiment.INFERENCE.R / alpha
    xg = model.xag.squeeze()

    def cost(x, u, a, x_lin):
            _J = pendulum_dobs(x_lin)
            _j = pendulum_obs(x_lin) - _J @ x_lin
            _x = _J @ x + _j
            return (_x - xg).T @ Q @ (_x - xg) + u.T @ R @ u

    # good values 100 iter 0.07
    gps = GuidedPolicySearch(
        model, cost, experiment.N_DURATION,
        kl_bound=0.07,
        u_lim=2,
        init_ctl_sigma=2.0,
        init_noise=1e-2,
        activation='all')

    cost = gps.run(
        100,
        )

    x, u, c = gps.forward_pass(gps.ctl)

    xu = np.vstack((x.mu[:,:-1], u.mu)).T
    env.plot_sim(xu, None, "gps pendulum internal", res_dir)

    cost = [c * alpha for c in cost]

    policy.K = np.zeros(policy.K.shape)
    policy.k = gps.udist.mu.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "gps pendulum feedforward", None)


    policy.K = gps.ctl.K.reshape(policy.K.shape)

    policy.k = gps.ctl.kff.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "gps pendulum", None)

    x = xu[:, :model.dim_x]
    u = xu[:, model.dim_x:]

    gps.plot_trajectory()

    f,a = plt.subplots(2)
    a[0].plot(cost, '.-')
    plt.savefig(os.path.join(res_dir, "cost_gps_pendulum.png"),
                bbox_inches='tight', format='png')

    x = gps.xdist.mu[:,:-1].T
    u = gps.udist.mu.T

    return x, u, cost, gps

def gps_cartpole(res_dir):
    experiment = cartpole_known
    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT,
                           experiment.MODEL)
    policy = TimeIndexedLinearGaussianPolicy(
        0 * np.eye(model.dim_u),
        experiment.N_DURATION, model.dim_u, model.dim_x)
    alpha = 1e3
    Q = experiment.INFERENCE.Q / alpha
    R = experiment.INFERENCE.R / alpha
    xg = model.xag.squeeze()

    def cost(x, u, a, x_lin):
        _J = cartpole_dobs(x_lin)
        _j = cartpole_obs(x_lin) - _J @ x_lin
        _x = _J @ x + _j
        return (_x - xg).T @ Q @ (_x - xg) + u.T @ R @ u

    gps = GuidedPolicySearch(
        model, cost, experiment.N_DURATION,
        kl_bound=1.0,
        u_lim=5,
        init_ctl_sigma=1.25,
        init_noise=1e-1,
        activation='all')

    cost = gps.run(
        200,
        )

    x, u, c = gps.forward_pass(gps.ctl)
    xu = np.vstack((x.mu[:,:-1], u.mu)).T
    env.plot_sim(xu, None, "gps cartpole internal", res_dir)

    cost = [c * alpha for c in cost]

    policy.K = np.zeros(policy.K.shape)
    policy.k = gps.udist.mu.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "gps cartpole feedforward", None)


    policy.K = gps.ctl.K.reshape(policy.K.shape)
    policy.k = gps.ctl.kff.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "gps cartpole", None)

    x = xu[:, :model.dim_x]
    u = xu[:, model.dim_x:]

    gps.plot_trajectory()

    f,a = plt.subplots(2)
    a[0].plot(cost, '.-')

    return x, u, cost, gps

def gps_double_cartpole(res_dir):
    experiment = double_cartpole_known
    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT,
                           experiment.MODEL)
    policy = TimeIndexedLinearGaussianPolicy(
        0 * np.eye(model.dim_u),
        experiment.N_DURATION, model.dim_u, model.dim_x)
    alpha = 1e3
    Q = experiment.INFERENCE.Q / alpha
    R = experiment.INFERENCE.R / alpha
    xg = model.xag.squeeze()

    def cost(x, u, a, x_lin):
        _J = double_cartpole_dobs(x_lin)
        _j = double_cartpole_obs(x_lin) - _J @ x_lin
        _x = _J @ x + _j
        return (_x - xg).T @ Q @ (_x - xg) + u.T @ R @ u
    gps = GuidedPolicySearch(
        model, cost, experiment.N_DURATION,
        kl_bound=0.75,
        u_lim=10,
        init_ctl_sigma=5,
        init_noise=1e-1,
        activation='all')
    cost = gps.run(
        200,
        )

    x, u, c = gps.forward_pass(gps.ctl)
    xu = np.vstack((x.mu[:,:-1], u.mu)).T
    env.plot_sim(xu, None, "gps  double cartpole internal", res_dir)

    cost = [c * alpha for c in cost]

    policy.K = np.zeros(policy.K.shape)
    policy.k = gps.udist.mu.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "gps double cartpole feedforward", None)


    policy.K = gps.ctl.K.reshape(policy.K.shape)
    policy.k = gps.ctl.kff.reshape(policy.k.shape)
    xu, dx, z = env.run(policy)
    env.plot_sim(xu, None, "gps double cartpole", None)

    x = xu[:, :model.dim_x]
    u = xu[:, model.dim_x:]

    gps.plot_trajectory()

    f,a = plt.subplots(2)
    a[0].plot(cost, '.-')

    return x, u, cost, gps

EXP = {
    "ilqr_pendulum": ilqr_pendulum,
    "ilqr_cartpole": ilqr_cartpole,
    "ilqr_double_cartpole": ilqr_double_cartpole,
    "gps_pendulum": gps_pendulum,
    "gps_cartpole": gps_cartpole,
    "gps_double_cartpole": gps_double_cartpole,
}

DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main(experiment, name):
    res_dir = os.path.join("_baselines", "%s_%s_%s" % (DATETIME, experiment, name))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    copyfile(__file__, os.path.join(res_dir, os.path.basename(__file__)))

    exp = EXP.get(experiment)
    x, u, cost, model = exp(res_dir)
    save_model(model, experiment, res_dir)

    # save data
    np.save(os.path.join(res_dir, "%s_x.npy" % experiment), x)
    np.save(os.path.join(res_dir, "%s_u.npy" % experiment), u)
    np.save(os.path.join(res_dir, "%s_cost.npy" % experiment), cost)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baselines")
    parser.add_argument("config",
                        help="baseline to run",
                        choices=EXP.keys())
    parser.add_argument("-n", "--name", help="helpful name for folder", default="")
    args = parser.parse_args()
    main(args.config, args.name)
    plt.show()
