
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib2tikz

from os.path import join

"""
TODO
load folder
folder as np files and config?
plot
"""

DIR_NAME = os.path.dirname(__file__)

try:
    import pi2c
except ImportError:

    print("pi2c not installed, using local version")
    top_path = os.path.join(DIR_NAME, '..')
    sys.path.append(os.path.abspath(top_path))

from pi2c.utils import configure_plots
import pi2c.env_def as env_def

def plot_traj(data, dim_x, dim_u, name, labels, dir_name=None):

    f,a = plt.subplots(dim_x + dim_u)
    a[0].set_title(name)
    for n, d in data.items():
        x,u = d
        for i in range(dim_x):
            a[i].plot(x[:, i], '.-', label=n)
        for j in range(dim_u):
            a[i+j+1].plot(u[:, j], '.-', label=n)

    for i,_a in enumerate(a):
        _a.set_ylabel(labels[i])

    a[-1].set_xlabel("Timesteps")
    for _a in a:
        _a.legend(loc="upper left")

    if dir_name is not None:
        name = name.replace(" ", "")
        plt.savefig(os.path.join(dir_name,
            "{}_traj.png".format(name)),
            bbox_inches='tight', format='png')
        matplotlib2tikz.save(
            os.path.join(dir_name, "{}_traj.tex".format(name)))
        # plt.close(f)

def plot_cost(data, name, dir_name=None):

    f = plt.figure()
    plt.title(name)
    for n, d in data.items():
        plt.plot(d, '.-', label=n)
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.legend()

    if dir_name is not None:
        name = name.replace(" ", "")
        plt.savefig(os.path.join(dir_name,
            "{}_cost.png".format(name)),
            bbox_inches='tight', format='png')
        matplotlib2tikz.save(
            os.path.join(dir_name, "{}_cost.tex".format(name)))
        # plt.close(f)

def plot_ctrl_perf(data, dir_name=None):
    for env, res in data.items():
        algos = res.keys()
        stats_cost = res.values()

        means = np.asarray([stat[0][0] for stat in stats_cost])
        stds = np.asarray([stat[0][1] for stat in stats_cost])
        final_cost = np.asarray([stat[1][-1] for stat in stats_cost])
        # means = [m / fc for m,fc in zip(means, final_cost)]
        # stds = [100 * s / fc for s,fc in zip(stds, final_cost)]
        for i, algo in enumerate(algos):
            print("{} {} {} {} +- {}".format(
                env, algo, final_cost[i], means[i], stds[i]))
        means = [m / fc for m,fc in zip(means, final_cost)]
        stds = [100 * s / fc for s,fc in zip(stds, final_cost)]
        for i, algo in enumerate(algos):
            print("{} {} {} +- {}".format(
                env, algo, means[i], stds[i]))
        fig, ax = plt.subplots()
        x_pos = np.arange(len(algos))
        ax.bar(x_pos, means, yerr=stds, align='center', color="white", edgecolor='k', linewidth=1)
        ax.set_ylabel('Controller Cost / Predicted Cost')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algos)
        ax.set_title(env)
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()

        if dir_name is not None:
            plt.savefig(os.path.join(dir_name,
                "{}_ctrl_perf.png".format(env)),
                bbox_inches='tight', format='png')
            matplotlib2tikz.save(
                os.path.join(dir_name, "{}_ctrl_perf.tex".format(env)))


def main():
    res_dir = "_plots"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    get_file = lambda name: join("scripts", "data", "%s.npy" %name)

    # costs
    # pendulum
    i2c_p = np.load(get_file("i2c_pendulum_cost"))
    ilqr_p = np.load(get_file("ilqr_pendulum_cost"))
    gps_p = np.load(get_file("gps_pendulum_cost"))
    # cartpole
    i2c_cp = np.load(get_file("i2c_cartpole_cost"))
    ilqr_cp = np.load(get_file("ilqr_cartpole_cost"))
    gps_cp = np.load(get_file("gps_cartpole_cost"))
    # double cartpole
    i2c_dcp = np.load(get_file("i2c_double_cartpole_cost"))
    ilqr_dcp = np.load(get_file("ilqr_double_cartpole_cost"))
    gps_dcp = np.load(get_file("gps_double_cartpole_cost"))

    # state action trajectories
    # pendulum
    i2c_p_x = np.load(get_file("i2c_pendulum_x"))
    i2c_p_u = np.load(get_file("i2c_pendulum_u"))
    ilqr_p_x = np.load(get_file("ilqr_pendulum_x"))
    ilqr_p_u = np.load(get_file("ilqr_pendulum_u"))
    gps_p_x = np.load(get_file("gps_pendulum_x"))
    gps_p_u = np.load(get_file("gps_pendulum_u"))

    i2c_cp_x = np.load(get_file("i2c_cartpole_x"))
    i2c_cp_u = np.load(get_file("i2c_cartpole_u"))
    ilqr_cp_x = np.load(get_file("ilqr_cartpole_x"))
    ilqr_cp_u = np.load(get_file("ilqr_cartpole_u"))
    gps_cp_x = np.load(get_file("gps_cartpole_x"))
    gps_cp_u = np.load(get_file("gps_cartpole_u"))

    i2c_dcp_x = np.load(get_file("i2c_double_cartpole_x"))
    i2c_dcp_u = np.load(get_file("i2c_double_cartpole_u"))
    ilqr_dcp_x = np.load(get_file("ilqr_double_cartpole_x"))
    ilqr_dcp_u = np.load(get_file("ilqr_double_cartpole_u"))
    gps_dcp_x = np.load(get_file("gps_double_cartpole_x"))
    gps_dcp_u = np.load(get_file("gps_double_cartpole_u"))

    i2c_p_ctrl = np.load(get_file("ctrl_i2c_pendulum"))
    gps_p_ctrl = np.load(get_file("ctrl_gps_pendulum"))
    ilqr_p_ctrl = np.load(get_file("ctrl_ilqr_pendulum"))

    i2c_cp_ctrl = np.load(get_file("ctrl_i2c_cartpole"))
    ilqr_cp_ctrl = np.load(get_file("ctrl_ilqr_cartpole"))
    gps_cp_ctrl = np.load(get_file("ctrl_gps_cartpole"))

    i2c_dcp_ctrl = np.load(get_file("ctrl_i2c_double_cartpole"))
    ilqr_dcp_ctrl = np.load(get_file("ctrl_ilqr_double_cartpole"))
    gps_dcp_ctrl = np.load(get_file("ctrl_gps_double_cartpole"))


    pen_traj = {"i2c": (i2c_p_x, i2c_p_u),
                "ilqr": (ilqr_p_x, ilqr_p_u),
                "gps":  (gps_p_x, gps_p_u)}

    cp_traj =   {"i2c": (i2c_cp_x, i2c_cp_u),
                "ilqr": (ilqr_cp_x, ilqr_cp_u),
                "gps":  (gps_cp_x, gps_cp_u)}

    dcp_traj =   {"i2c": (i2c_dcp_x, i2c_dcp_u),
                "ilqr": (ilqr_dcp_x, ilqr_dcp_u),
                "gps":  (gps_dcp_x, gps_dcp_u)
            }

    pen = {"i2c": i2c_p, "ilqr": ilqr_p, "gps": gps_p}
    cp  = {"i2c": i2c_cp, "ilqr": ilqr_cp, "gps": gps_cp}
    dcp = {"i2c": i2c_dcp, "ilqr": ilqr_dcp, "gps": gps_dcp}

    ctrl = {"Pendulum": {"i2c": (i2c_p_ctrl, i2c_p),
                         "gps": (gps_p_ctrl, gps_p),
                         "ilqr": (ilqr_p_ctrl, ilqr_p)
                         },
            "Cartpole": {"i2c": (i2c_cp_ctrl, i2c_cp),
                         "ilqr": (ilqr_cp_ctrl, ilqr_cp),
                         "gps":  (gps_cp_ctrl, gps_cp)},
            "Double Cartpole": {"i2c": (i2c_dcp_ctrl, i2c_dcp),
                                "ilqr": (ilqr_dcp_ctrl, ilqr_dcp),
                                "gps": (gps_dcp_ctrl, gps_dcp)}
            }

    plot_cost(pen, "Pendulum", res_dir)
    plot_cost(cp, "Cartpole", res_dir)
    plot_cost(dcp, "Double Cartpole", res_dir)

    plot_traj(pen_traj, 2, 1, "Pendulum",
        env_def.PendulumKnown.key,
        res_dir)

    plot_traj(cp_traj, 4, 1, "Cartpole",
        env_def.CartpoleKnown.key,
        res_dir)

    plot_traj(dcp_traj, 6, 1, "Double Cartpole",
        env_def.DoubleCartpoleKnown.key,
        res_dir)

    plot_ctrl_perf(ctrl, res_dir)


if __name__ == "__main__":
    # configure_plots()
    main()
    plt.show()