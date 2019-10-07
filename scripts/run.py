"""
i2c Experiment Runner
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import argparse
import importlib
import logging
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar
from shutil import copyfile

DIR_NAME = os.path.dirname(__file__)

try:
    import pi2c
except ImportError:
    print("pi2c not installed, using local version")
    top_path = os.path.join(DIR_NAME, '..')
    sys.path.append(os.path.abspath(top_path))

from pi2c.i2c import I2cGraph
from pi2c.env import make_env
from pi2c.model import make_env_model
from pi2c.policy.linear import TimeIndexedLinearGaussianPolicy
from pi2c.utils import converged_list, TrajectoryData, profile, make_results_folder, configure_plots, TrajectoryEvaluator

PROFILING = False

def run(experiment, res_dir, weight_path):

    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT,
                           experiment.MODEL)
    i2c = I2cGraph(
        model,
        experiment.N_DURATION,
        experiment.INFERENCE.Q,
        experiment.INFERENCE.R,
        experiment.INFERENCE.ALPHA,
        experiment.INFERENCE.alpha_update_tol,
        experiment.INFERENCE.SIG_U,
        experiment.INFERENCE.msg_iter,
        experiment.INFERENCE.msg_tol,
        experiment.INFERENCE.em_tol,
        experiment.INFERENCE.backwards_contraction,
        res_dir)

    if weight_path is not None:
        print("Loading i2c model with {}".format(weight_path))
        i2c.sys.model.load(weight_path)

    policy = TimeIndexedLinearGaussianPolicy(experiment.POLICY_COVAR,
        experiment.N_DURATION, i2c.sys.dim_u, i2c.sys.dim_x)

    # collection of data
    traj_data = TrajectoryData(
        model.x_noise, model.y_noise, experiment.N_AUG)
    s_est = np.zeros((experiment.N_DURATION, i2c.sys.dim_xt))
    traj_eval = TrajectoryEvaluator(experiment.N_DURATION, i2c.QR, i2c.sg)
    traj_eval_iter = TrajectoryEvaluator(experiment.N_DURATION, i2c.QR, i2c.sg)

    # get stationary data for initial training data
    for _ in range(experiment.N_STARTING):
        x, y, _ = env.run(policy)
        traj_data.add(x, y)

    # MBRL training loop
    for j in range(experiment.N_EPISODE):
        print("Ep. {}".format(j))

        # run policy on env, get data
        with profile("Simulation", PROFILING):
            x, y, _ = env.run(policy)
            x_test, y_test, _ = env.run(policy)
            env.plot_sim(x, s_est, "training {}".format(j), res_dir)

        # fit model
        x_train, y_train = traj_data.add(x, y)

        # inference
        bar = Bar('Learning', max=experiment.N_INFERENCE)
        with profile("Inference", PROFILING):
            i2c.reset_metrics()
            for i in range(experiment.N_INFERENCE):
                i2c.learn_msgs()
                # eval policy
                policy.K, policy.k, _ = i2c.get_local_linear_policy()
                policy.sigk = np.zeros(policy.sigk.shape)
                x, y, z = env.run(policy)
                z_est = i2c.get_marginal_observed_trajectory()
                traj_eval_iter.eval(z, z_est)

                if  i % experiment.N_ITERS_PER_PLOT == 0: #
                    i2c.plot_traj(i, dir_name=res_dir, filename="iter_{}_{}".format(j, i))
                    i2c.plot_observed_traj(dir_name=res_dir, filename="iter_{}_{}".format(j, i))
                    i2c.plot_system_dynamics(dir_name=res_dir, filename="iter_{}_{}".format(j, i))
                    i2c.plot_uncertainty(dir_name=res_dir, filename="iter_{}_{}".format(j, i))
                    i2c.plot_controller(dir_name=res_dir, filename="{}_{}".format(j, i))
                    i2c.plot_cost(res_dir, j)
                    i2c.plot_alphas(dir_name=res_dir, filename=j)
                    i2c.plot_policy_entropy(dir_name=res_dir, filename=j)
                    policy.K, policy.k, policy.sigk = i2c.get_local_linear_policy()
                    s_est = i2c.get_marginal_trajectory()
                    env.plot_sim(x, s_est, "{} {}".format(j, i), res_dir)
                    i2c.plot_gap(res_dir, j)
                    i2c.plot_em_cost(res_dir, j)
                    traj_eval_iter.plot("over_iterations_{}".format(j), res_dir)
                    i2c.save(res_dir, "{}_{}".format(j, i))
                bar.next()
            bar.finish()

        i2c.plot_traj(i, dir_name=res_dir, filename="iter_{}_{}".format(j, i))
        i2c.plot_cost_all(res_dir, j)
        i2c.plot_uncertainty(dir_name=res_dir, filename=j)
        i2c.plot_observed_traj(dir_name=res_dir, filename="iter_{}_{}".format(j, i))
        i2c.plot_controller(dir_name=res_dir, filename=j)

        # update policy
        policy.K, policy.k, policy.sigk = i2c.get_local_linear_policy()
        z_est = i2c.get_marginal_observed_trajectory()
        x, y, z = env.run(policy)
        s_est = i2c.get_marginal_trajectory()
        env.plot_sim(x, s_est, "evaluation {}".format(j), res_dir)
        traj_eval.eval(z, z_est)
        traj_eval.plot("over_episodes", res_dir)
        traj_eval_iter.plot("over_iterations_{}".format(j), res_dir)

    i2c.plot_alphas(res_dir, "final")
    i2c.plot_cost(res_dir, "cost_final")

    x_final, _, _ = env.run(policy)
    s_est = i2c.get_marginal_trajectory()
    env.plot_sim(x_final, s_est, "Final", res_dir)

    # compare against pure feedforward
    policy.zero()
    policy.k = i2c.get_marginal_input()
    x_ff, _, _ = env.run(policy)
    env.plot_sim(x_ff, s_est, "Final Feedforward", res_dir)

    # save model and data
    i2c.sys.save(res_dir)
    save_trajectories(x_final, i2c, res_dir)
    traj_eval.save("episodic", res_dir)
    traj_eval_iter.save("iter", res_dir)
    i2c.save(res_dir, "{}_{}".format(j, i))

    env.close()


def save_trajectories(s, inference, res_dir):
    if res_dir:
        x_real = s[:, :inference.sys.dim_x]
        u_real = s[:, inference.sys.dim_x:]
        np.save(os.path.join(res_dir, "xu_real.npy"), s)
        np.save(os.path.join(res_dir, "x_real.npy"), x_real)
        np.save(os.path.join(res_dir, "u_real.npy"), u_real)
        inference.save_traj(res_dir)

if __name__ == "__main__":
    import git
    from glob import glob
    from os.path import basename, join, splitext

    def write_commit(res_dir):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        with open(os.path.join(res_dir, "git_commit.txt"), 'w+') as f:
            f.write(sha)

    configure_plots()

    exps = [splitext(basename(e))[0] for e in
            glob(os.path.join(DIR_NAME, "experiments", "*.py"))
            if "__init__" not in e]

    parser = argparse.ArgumentParser(description="Run Experiment")
    parser.add_argument("config",
                        help="file with hyperparameters",
                        choices=exps)
    parser.add_argument("-n", "--name", help="folder", default="")
    parser.add_argument("-w", "--weights", help="path to model weights")
    parser.add_argument("-r", "--random-seed", help="random seed", default=0)
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    experiment = importlib.import_module(
        "experiments.{}".format(args.config))
    exp_filename = "{}.py".format(args.config)
    exp_file = os.path.join(os.path.join(DIR_NAME,
        os.path.join("experiments", exp_filename)))
    res_dir = make_results_folder(args.config, args.random_seed, args.name)
    # copy experiment config and git commit to results
    copyfile(exp_file, os.path.join(res_dir, exp_filename))
    write_commit(res_dir)

    run(experiment, res_dir, args.weights)
    plt.show()
