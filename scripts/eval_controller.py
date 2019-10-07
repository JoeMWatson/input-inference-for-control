
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
from pi2c.utils import TrajectoryEvaluator

from baselines.gps import GuidedPolicySearch
from baselines.ilqr import IterativeLqr

DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

N_SAMPLES = 100
ALPHA = float(10 / N_SAMPLES)

def load_algo(path):
    with open(path, 'rb') as f:
        obj = dill.load(f)
    return obj

def i2c_getter(i2c):
    return i2c.get_local_linear_policy()[0:2]

def gps_getter(gps):
    K = gps.ctl.K
    k = gps.ctl.kff
    u = gps.udist.mu
    x = gps.xdist.mu[:, :-1]
    dim_u, dim_x, h = K.shape
    Kx = np.einsum("jki,ki->ji", K, x).reshape((h, dim_u))
    u = u.reshape((h, dim_u))
    K = K.reshape((h, dim_u, dim_x))
    k = k.reshape((h, dim_u))
    x = x.reshape((h, dim_x))
    plt.figure()
    plt.plot(u, label="u")
    plt.plot(k, label="k")
    plt.plot(Kx, label="Kx")
    plt.legend()
    return K, k

def ilqr_getter(ilqr):
    K = ilqr.ctl.K
    u = ilqr.uref
    x = ilqr.xref[:, :-1]
    dim_u, dim_x, h = K.shape
    Kx = np.einsum("jki,ki->ji", K, x).reshape((h, dim_u))
    plt.figure()
    plt.plot(K.squeeze().T)
    plt.figure()
    plt.plot(x.T)
    u = u.reshape((h, dim_u))
    K = K.reshape((h, dim_u, dim_x))
    k = u - Kx
    plt.figure()
    plt.plot(u, label="u")
    plt.plot(k, label="k")
    plt.plot(Kx, label="Kx")
    plt.legend()
    return K, k

def i2c_traj_getter(i2c):
    return i2c.get_state_and_action()

def gps_traj_getter(gps):
    x = gps.xdist.mu[:, :-1].T
    u = gps.udist.mu.T
    f,a = plt.subplots(2)
    a[0].plot(x)
    a[1].plot(u)
    return x,u

def ilqr_traj_getter(ilqr):
    x = ilqr.xref[:, :-1].T
    u = ilqr.uref.T
    f,a = plt.subplots(2)
    a[0].plot(x)
    a[1].plot(u)
    return x,u

CTRL_GET = {"i2c": i2c_getter,
            "gps": gps_getter,
            "ilqr":  ilqr_getter}

TRAJ_GET = {"i2c": i2c_traj_getter,
            "gps": gps_traj_getter,
            "ilqr":  ilqr_traj_getter}

EXPS = {"pendulum": pendulum_known,
        "cartpole": cartpole_known,
        "double_cartpole": double_cartpole_known}

def main(path, algo_name, env_name, name):
    res_dir = os.path.join("_baselines", "%s_ctlr_%s_%s_%s" % (DATETIME, algo_name, env_name, name))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    exp = EXPS.get(env_name)
    trajectory_getter = TRAJ_GET.get(algo_name)
    control_getter = CTRL_GET.get(algo_name)
    env = make_env(exp)
    # env.sigV = 10 * env.sigV
    policy = TimeIndexedLinearGaussianPolicy(
        0 * np.eye(env.dim_u),
        exp.N_DURATION, env.dim_u, env.dim_x)
    QR = np.block([[exp.INFERENCE.Q,                   np.zeros((env.dim_xa, env.dim_u))],
                   [np.zeros((env.dim_u, env.dim_xa)), exp.INFERENCE.R]])
    traj_eval = TrajectoryEvaluator(exp.N_DURATION,
        QR, env.sg)

    algo = load_algo(path)
    policy.K, policy.k = control_getter(algo)
    x, u = trajectory_getter(algo)
    filepath = os.path.join(res_dir, "%s_%s_x.npy" % (algo_name, env_name))
    np.save(filepath, x)
    filepath = os.path.join(res_dir, "%s_%s_u.npy" % (algo_name, env_name))
    np.save(filepath, u)

    f,a = plt.subplots(env.dim_y)
    for _a, l in zip(a, env.y_key):
        _a.set_ylabel(l)
    for _ in range(N_SAMPLES):
        _, _, z = env.run(policy)
        for i in range(env.dim_y):
            a[i].plot(z[:, i], '.-', alpha=ALPHA)
        traj_eval.eval(z, z) # use both whatevs

    cost = np.asarray(traj_eval.actual_cost)
    mean = np.mean(cost)
    std = np.std(cost)
    print("mean", mean, "std", std)
    filepath = os.path.join(res_dir, "ctrl_%s_%s.npy" % (algo_name, env_name))
    np.save(filepath, np.asarray([mean, std]))
    plt.savefig(os.path.join(res_dir, "%s_%s.png" % (algo_name, env_name)),
        bbox_inches='tight', format='png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Controller Evaluation")
    parser.add_argument("path",
                        help="path to pickled instance")
    parser.add_argument("algo", help="algo type", choices=["i2c", "gps", "ilqr"])
    parser.add_argument("env", help="env type", choices=["pendulum", "cartpole", "double_cartpole"])
    parser.add_argument("-n", "--name", help="helpful name for folder", default="")
    args = parser.parse_args()
    main(args.path, args.algo, args.env, args.name)
    plt.show()