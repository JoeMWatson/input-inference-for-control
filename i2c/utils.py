""""
Misc. helper code, ranging from refence optimal control solvers to matplotlib config.
"""
import datetime
import numpy as np
import os


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import logging


DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def quadratic_trajectory_cost(z, z_term, zg, zg_term, QR, Qf):
    """Compute quadratic cost from trajectory"""
    assert z.shape[1] == QR.shape[0] == zg.shape[0]
    assert z_term.shape[1] == Qf.shape[0] == zg_term.shape[0]
    assert QR.shape[0] == QR.shape[1]
    assert Qf.shape[0] == Qf.shape[1]
    err = z - zg.reshape((1, -1))
    err_term = z_term - zg_term.reshape((1, -1))
    return np.einsum("bi,ij,bj->", err, QR, err) + np.asscalar(
        err_term @ Qf @ err_term.T
    )


def finite_horizon_lqr_tv(H, A, a, B, Q, R, Qf, q, r, qf, x0, dim_x, dim_u):

    K = np.zeros((H, dim_u, dim_x))
    k = np.zeros((H, dim_u))
    Ps = np.zeros((H, dim_x, dim_x))
    ps = np.zeros((H, dim_x))
    P = Qf
    p = -qf
    for i in range(H - 1, -1, -1):
        _A = A[i, :, :]
        _a = a[i, :, :]
        _B = B[i, :, :]
        _Q = Q[i, :, :]
        _q = q[i, :, :]
        _R = R[i, :, :]
        _r = r[i, :, :]
        Ps[i, :, :] = P.squeeze()
        ps[i, :] = p.squeeze()
        _M = _R + _B.T @ P @ _B
        _Minv = np.linalg.inv(_M)
        _K = np.linalg.inv(_M) @ _B.T
        K[i, :, :] = -_Minv @ _B.T @ P @ _A
        k[i, :] = -_Minv @ (_B.T @ P @ _a + _B.T @ p - _r)
        _P = _Q + _A.T @ P @ _A - _A.T @ P @ _B @ _Minv @ _B.T @ P @ _A
        p = _A.T @ (P @ _a + p - P @ _B @ _Minv @ (_B.T @ (P @ _a + p) - _r)) - _q
        P = _P
    return K, k


def finite_horizon_lqr(H, A, a, B, Q, R, x0, xg, ug, dim_x, dim_u):
    a = a
    xg = xg
    K = np.zeros((H, dim_u, dim_x))
    k = np.zeros((H, dim_u))
    Ps = np.zeros((H, dim_x, dim_x))
    ps = np.zeros((H, dim_x))
    P = Q
    p = -Q @ xg
    for i in range(H - 1, -1, -1):
        Ps[i, :, :] = P.squeeze()
        ps[i, :] = p.squeeze()
        _M = R + B.T @ P @ B
        _Minv = np.linalg.inv(_M)
        _K = np.linalg.inv(_M) @ B.T
        K[i, :, :] = -_Minv @ B.T @ P @ A
        k[i, :] = -_Minv @ (B.T @ P @ a + B.T @ p - R @ ug)
        _P = Q + A.T @ P @ A - A.T @ P @ B @ _Minv @ B.T @ P @ A
        p = A.T @ (P @ a + p - P @ B @ _Minv @ (B.T @ (P @ a + p) - R @ ug)) - Q @ xg
        P = _P

    x_lqr = np.zeros((H, dim_x))
    u_lqr = np.zeros((H, dim_u))
    u_K = np.zeros((H, dim_u))
    u_k = np.zeros((H, dim_u))
    x = x0
    cost = 0.0
    for i in range(H):
        x_lqr[i, :] = x
        u_K[i] = K[i, :, :] @ x
        u_k[i] = k[i, :]
        u_lqr[i] = u_K[i] + u_k[i]
        u = u_lqr[i]
        e_x = x - xg
        e_u = u - ug
        c = e_x.T @ Q @ e_x + e_u.T @ R @ e_u
        cost += c
        x = A @ x + B @ u_lqr[i] + a
    e_x = x - xg
    cost += e_x.T @ Q @ e_x

    return x_lqr, u_lqr, K, k, cost, Ps, ps


class TrajectoryEvaluator(object):
    def __init__(self, W, Wf, sg, sg_term, dim_x):
        self.W = W
        self.Wf = Wf
        self.sg, self.sg_term = sg.reshape(-1, 1), sg_term.reshape(-1, 1)
        self.actual_cost = []
        self.planned_cost = []
        self.dim_x = dim_x

        d = W.shape[0]
        assert self.W.shape[1] == d
        assert self.sg.shape[0] == d

    def _eval_traj(self, s, s_terminal):
        error = s - self.sg.T
        error_t = (s_terminal - self.sg_term.T)[-1, : self.dim_x]
        traj = np.einsum("bi,ij,bj->", error[:-1, :], self.W, error[:-1, :])
        terminal = np.asscalar(error_t.T.dot(self.Wf.dot(error_t)))
        return traj + terminal

    def eval(self, actual_traj, actual_terminal, planned_traj, planned_terminal):
        self.actual_cost.append(self._eval_traj(actual_traj, actual_terminal))
        self.planned_cost.append(self._eval_traj(planned_traj, planned_terminal))

    def plot(self, name, res_dir=None):
        f = plt.figure()
        plt.title("Trajectory Cost over evaluations ")
        plt.plot(self.actual_cost, "ro-", label="Actual")
        plt.plot(self.planned_cost, "bo-", label="Planned")
        plt.legend()
        plt.xlabel("Evaluations")
        plt.ylabel("Cost")
        if res_dir is not None:
            plt.savefig(
                os.path.join(res_dir, "traj_eval_{}.png".format(name)),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def save(self, name, res_dir):
        actual = np.asarray(self.actual_cost)
        plan = np.asarray(self.planned_cost)
        np.save(os.path.join(res_dir, "cost_actual_{}.npy".format(name)), actual)
        np.save(os.path.join(res_dir, "cost_plan_{}.npy".format(name)), plan)


class StochasticTrajectoryEvaluator(object):
    def __init__(self, W, Wf, sg, sg_term, dim_x):
        self.W = W
        self.Wf = Wf
        self.sg, self.sg_term = sg.reshape((-1, 1)), sg_term.reshape((-1, 1))
        self.mu_actual_cost = []
        self.max_actual_cost = []
        self.min_actual_cost = []
        self.actual_cost_10 = []
        self.actual_cost_90 = []
        self.planned_cost = []
        self.dim_x = dim_x

        self.dim_s = W.shape[0]
        assert self.W.shape[1] == self.dim_s
        assert self.sg.shape[0] == self.dim_s, f"{self.sg.shape[0]} ~= {self.dim_s}"
        # assert self.Wf.shape == (dim_x, dim_x)

    def _eval_traj(self, s, s_term):
        error = s - self.sg.reshape((-1, self.dim_s))
        traj = np.einsum("bi,ij,bj->b", error[:-1, :], self.W, error[:-1, :]).sum(0)
        if s_term is not None:
            error_t = (s_term.reshape((1, -1)) - self.sg_term.reshape((1, -1)))[
                -1, : self.dim_x
            ]  # we only care about x
            traj += error_t.dot(self.Wf.dot(error_t.T))
        return traj

    def eval(self, actual_trajs, actual_trajs_term, planned_traj, planned_traj_term):
        actual_costs = np.asarray(
            [
                self._eval_traj(traj, term)
                for traj, term in zip(actual_trajs, actual_trajs_term)
            ]
        )

        self.mu_actual_cost.append(np.mean(actual_costs))
        self.min_actual_cost.append(np.min(actual_costs))
        self.max_actual_cost.append(np.max(actual_costs))
        p10, p90 = np.percentile(actual_costs, (10, 90))
        self.actual_cost_10.append(p10)
        self.actual_cost_90.append(p90)
        self.planned_cost.append(self._eval_traj(planned_traj, planned_traj_term))

    def plot(self, name, res_dir=None):
        f, ax = plt.subplots(1, 2)
        n = len(self.planned_cost)
        t = range(n)
        upper = np.asarray(self.max_actual_cost).squeeze()
        lower = np.asarray(self.min_actual_cost).squeeze()
        p10 = np.asarray(self.actual_cost_10).squeeze()
        p90 = np.asarray(self.actual_cost_90).squeeze()
        a = ax[0]
        plt.title("Trajectory Cost over evaluations ")
        a.fill_between(t, upper, lower, where=upper >= lower, facecolor="r", alpha=0.2)
        a.fill_between(t, p90, p10, where=p90 >= p10, facecolor="r", alpha=0.4)
        a.plot(upper, "r+-", label="Actual Cost Max")
        a.plot(lower, "r^-", label="Actual Cost Min")
        a.plot(p90, "rx-", label="Actual Cost 90th")
        a.plot(p10, "r*-", label="Actual Cost 10th")
        a.plot(self.mu_actual_cost, "ro-", label="Actual Cost Mean")
        a.plot(self.planned_cost, "bo-", label="Planned")
        a.legend()
        a.set_xlabel("Evaluations")
        a.set_ylabel("Cost")

        a = ax[1]
        a.set_xlabel("Evaluations")
        a.set_ylabel("Cost Error")
        a.plot(np.abs(np.asarray(self.mu_actual_cost) - np.asarray(self.planned_cost)))
        if res_dir is not None:
            plt.savefig(
                os.path.join(res_dir, "traj_eval_{}.png".format(name)),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)
        else:
            return f

    def plot_sample(self, name, res_dir=None):
        f = plt.figure()
        n = len(self.planned_cost)
        t = range(n)
        upper = np.asarray(self.max_actual_cost).squeeze()
        lower = np.asarray(self.min_actual_cost).squeeze()
        p10 = np.asarray(self.actual_cost_10).squeeze()
        p90 = np.asarray(self.actual_cost_90).squeeze()
        plt.title("Trajectory Cost over evaluations ")
        plt.fill_between(
            t, upper, lower, where=upper >= lower, facecolor="r", alpha=0.2
        )
        plt.fill_between(t, p90, p10, where=p90 >= p10, facecolor="r", alpha=0.4)
        plt.plot(upper, "r+-", label="Actual Cost Max")
        plt.plot(lower, "r^-", label="Actual Cost Min")
        plt.plot(p90, "rx-", label="Actual Cost 90th")
        plt.plot(p10, "r*-", label="Actual Cost 10th")
        plt.plot(self.mu_actual_cost, "ro-", label="Actual Cost Mean")
        plt.legend()
        plt.xlabel("Evaluations")
        plt.ylabel("Cost")
        if res_dir is not None:
            plt.savefig(
                os.path.join(res_dir, "traj_eval_{}.png".format(name)),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)
        else:
            return f

    def save(self, name, res_dir):
        actual = np.asarray(self.mu_actual_cost)
        plan = np.asarray(self.planned_cost)
        np.save(os.path.join(res_dir, "cost_actual_mean_{}.npy".format(name)), actual)
        np.save(os.path.join(res_dir, "cost_plan_{}.npy".format(name)), plan)


class TrajectoryEvaluatorEnsemble(object):
    def __init__(self, horizon, W, Wg, sg, n_ensemble, n_samp):
        self.horizon = horizon
        self.W = W
        self.Wg = Wg
        self.n_ensemble = n_ensemble
        self.n_samp = n_samp
        self.sg = sg.reshape(-1, 1)
        self.ensemble_cost = []  # of (n_ensembl,) arrays
        self.sample_cost = []  # of (mean, std) tuples

        self.d = W.shape[0]
        assert self.W.shape[1] == self.d
        assert self.sg.shape[0] == self.d

    def _eval_traj(self, s):
        error = s - self.sg.T
        error_t = error[-1, : self.dim_x]  # we only care about x
        traj = np.einsum("bi,ij,bj->b", error[:-1, :], self.W, error[:-1, :])
        terminal = error_t.T.dot(self.Wf.dot(error_t))
        return traj.sum(0) + terminal

    def eval_ensemble(self, ensemble_trajs):
        costs = np.asarray([self._eval_traj(traj) for traj in ensemble_trajs]).squeeze()
        self.ensemble_cost.append(costs)

    def eval_sample(self, sample_trajs):
        costs = np.asarray([self._eval_traj(traj) for traj in sample_trajs]).squeeze()
        mean = np.mean(costs)
        std = np.std(costs)
        maxi = np.max(costs)
        mini = np.min(costs)
        p99 = np.percentile(costs, 0.99)
        self.sample_cost.append((mean, std, maxi, mini, p99, costs))

    def plot(self, name, res_dir=None):
        n = len(self.sample_cost)
        ensemble = np.asarray(self.ensemble_cost).reshape((n, self.n_ensemble))
        m_sample = np.asarray([v[0] for v in self.sample_cost])
        std_sample = np.asarray([v[1] for v in self.sample_cost])
        max_sample = np.asarray([v[2] for v in self.sample_cost])
        min_sample = np.asarray([v[3] for v in self.sample_cost])
        p99_sample = np.asarray([v[4] for v in self.sample_cost])
        sample = np.vstack([v[5] for v in self.sample_cost])
        sample_upper = m_sample + 2 * std_sample
        sample_lower = m_sample - 2 * std_sample
        f = plt.figure()
        plt.title("Ensemble Trajectory Cost over evaluations ")
        for i in range(self.n_ensemble):
            plt.plot(ensemble[:, i], "ro-", label="Ensemble" if i == 0 else None)
        plt.fill_between(
            range(n),
            sample_lower,
            sample_upper,
            where=sample_upper > sample_lower,
            color="b",
            alpha=0.3,
        )
        plt.plot(sample, "b+", alpha=0.3)
        plt.plot(max_sample, "bx--", label="Sample Max")
        plt.plot(min_sample, "bx--", label="Sample Min")
        plt.plot(m_sample, "co-", label="Sample Mean")
        plt.plot(p99_sample, "m^-", label="Sample 99th percentile")
        plt.ylim(0, 1.1 * np.max(m_sample))
        plt.legend()
        plt.grid()
        plt.xlabel("Evaluations")
        plt.ylabel("Cost")
        if res_dir is not None:
            plt.savefig(
                os.path.join(res_dir, "traj_eval_{}.png".format(name)),
                bbox_inches="tight",
                format="png",
            )
            plt.close(f)

    def save(self, name, res_dir):
        actual = np.asarray(self.actual_cost)
        plan = np.asarray(self.planned_cost)
        np.save(os.path.join(res_dir, "cost_actual_{}.npy".format(name)), actual)
        np.save(os.path.join(res_dir, "cost_plan_{}.npy".format(name)), plan)


def set_seed(seed):
    np.random.seed(seed)


def make_results_folder(config, seed, name, folder_name="_results", release=False):
    components = [config.replace(" ", "-"), str(seed), name.replace(" ", "-")]
    if release:
        components = ["release"] + components
    else:
        components = [DATETIME] + components
    folder = "_".join(components)
    res_dir = os.path.join(folder_name, folder)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    return res_dir


def configure_plots():
    import matplotlib

    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["figure.figsize"] = [16, 16]
    matplotlib.rcParams["legend.fontsize"] = 16
    matplotlib.rcParams["axes.titlesize"] = 22
    matplotlib.rcParams["axes.labelsize"] = 22


def covariance_2d(covar, mean, axis, n_std=2.0, facecolor="b", **kwargs):

    w, v = np.linalg.eig(covar)
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
        **kwargs,
    )

    return axis.add_patch(ellipse)


def setup_logger(res_dir, level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(res_dir, "output.log"),
        level=level,
        format="[%(asctime)s] %(pathname)s:%(lineno)d %(levelname)s" "- %(message)s",
    )


def plot_uncertainty(ax, x, mean, variance, stds=[2], color="b", alpha=0.1):
    x = x.squeeze()
    mean = mean.squeeze()
    variance = variance.squeeze()
    for sf in stds:
        y_std = sf * np.sqrt(variance)
        y_upper = mean + y_std
        y_lower = mean - y_std
        ax.fill_between(
            x, y_upper, y_lower, where=y_upper > y_lower, color=color, alpha=alpha
        )


def write_commit(res_dir):
    import git

    # TODO check git exists?
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(res_dir, "git_commit.txt"), "w+") as f:
        f.write(sha)
