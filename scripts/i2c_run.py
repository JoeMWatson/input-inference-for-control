"""
i2c Trajectory Optimization Experiment Runner
"""
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from shutil import copyfile

from i2c.i2c import I2cGraph
from i2c.env import make_env
from i2c.model import make_env_model
from i2c.policy.linear import (
    TimeIndexedLinearGaussianPolicy,
    ExpertTimeIndexedLinearGaussianPolicy,
)
from i2c.utils import StochasticTrajectoryEvaluator
from i2c.utils import (
    make_results_folder,
    configure_plots,
    set_seed,
    setup_logger,
)

N_EVAL = 10


def run(experiment, res_dir, weight_path):
    env = make_env(experiment)
    model = make_env_model(experiment.ENVIRONMENT, experiment.MODEL)

    i2c = I2cGraph(
        model,
        experiment.N_DURATION,
        experiment.INFERENCE.Q,
        experiment.INFERENCE.R,
        experiment.INFERENCE.Qf,
        experiment.INFERENCE.alpha,
        experiment.INFERENCE.alpha_update_tol,
        experiment.INFERENCE.mu_u,
        experiment.INFERENCE.sig_u,
        experiment.INFERENCE.mu_x_term,
        experiment.INFERENCE.sig_x_term,
        experiment.INFERENCE.inference,
        res_dir=res_dir,
    )

    policy_class = ExpertTimeIndexedLinearGaussianPolicy
    policy_linear = TimeIndexedLinearGaussianPolicy(
        experiment.POLICY_COVAR, experiment.N_DURATION, i2c.sys.dim_u, i2c.sys.dim_x
    )
    policy = policy_class(
        experiment.POLICY_COVAR,
        experiment.N_DURATION,
        i2c.sys.dim_u,
        i2c.sys.dim_x,
        soft=False,
    )

    if weight_path is not None:
        print("Loading i2c model with {}".format(weight_path))
        i2c.sys.model.load(weight_path)

    # initial marginal traj
    s_est = np.zeros((experiment.N_DURATION, model.dim_s))

    dim_terminal = i2c.Qf.shape[0]
    traj_eval = StochasticTrajectoryEvaluator(
        i2c.QR, i2c.Qf, i2c.z, i2c.z_term, dim_terminal
    )
    traj_eval_iter = StochasticTrajectoryEvaluator(
        i2c.QR, i2c.Qf, i2c.z, i2c.z_term, dim_terminal
    )
    traj_eval_safe_iter = StochasticTrajectoryEvaluator(
        i2c.QR, i2c.Qf, i2c.z, i2c.z_term, dim_terminal
    )

    i2c.reset_metrics()

    if env.simulated:
        policy.zero()
        xs, ys, zs, z_term = env.batch_eval(policy, N_EVAL)
        env.plot_sim(xs, s_est, "initial", res_dir)
        traj_eval.eval(zs, z_term, zs[0], z_term[0])

    # inference
    try:
        for i in tqdm(range(experiment.N_INFERENCE)):
            plot = (i % experiment.N_ITERS_PER_PLOT == 0) or (
                i == experiment.N_INFERENCE - 1
            )

            i2c.learn_msgs()

            if env.simulated:
                # eval policy
                policy_linear.write(*i2c.get_local_linear_policy())

                xs, ys, zs, zs_term = env.batch_eval(policy_linear, N_EVAL)
                z_est, z_term_est = i2c.get_marginal_observed_trajectory()
                traj_eval_iter.eval(zs, zs_term, z_est, z_term_est)

                policy.write(*i2c.get_local_expert_linear_policy())
                xs, ys, zs, zs_term = env.batch_eval(policy, N_EVAL)
                traj_eval_safe_iter.eval(zs, zs_term, z_est, z_term_est)

                logging.info(
                    f"{i:02d} Cost | Plan: {i2c.costs_m[-1]}, "
                    f"Predict: {i2c.costs_pf[-1]}, "
                    f"Sim: [{traj_eval_iter.actual_cost_10[-1]}, "
                    f"{traj_eval_iter.actual_cost_90[-1]}] "
                    f"alpha: {i2c.alphas[-1], i2c.alphas_desired[-1]}"
                )

            if i == 0:  # see how well inference works at the start
                xs, ys, zs, zs_term = env.batch_eval(
                    policy, N_EVAL, deterministic=False
                )
                env.plot_sim(xs, s_est, f"{i}_stochastic", res_dir)

            if plot:
                i2c.plot_metrics(0, i, res_dir, "msg")
                s_est = i2c.get_marginal_trajectory()
                env.plot_sim(xs, s_est, f"{i}_stochastic", res_dir)

        i2c.plot_metrics(0, i, res_dir, "msg")
    except Exception as ex:
        logging.exception("Inference failed")
        i2c.plot_metrics(0, i, res_dir, "esc")
        raise

    # update policy
    if env.simulated:
        # policy.write(*i2c.get_local_linear_policy())
        policy_linear.write(*i2c.get_local_linear_policy())
        z_est, z_term_est = i2c.get_marginal_observed_trajectory()
        xs, ys, zs, zs_term = env.batch_eval(policy_linear, N_EVAL)
        s_est = i2c.get_marginal_trajectory()
        env.plot_sim(xs, s_est, f"evaluation stochastic", res_dir)

        xs, ys, zs, zs_term = env.batch_eval(policy_linear, N_EVAL)
        env.plot_sim(xs, s_est, f"evaluation deterministic", res_dir)

        z_est, z_term_est = i2c.get_marginal_observed_trajectory()
        traj_eval_iter.eval(zs, zs_term, z_est, z_term_est)
        traj_eval.eval(zs, zs_term, z_est, z_term_est)
        traj_eval_iter.plot("over_iterations", res_dir)
        traj_eval.plot("over_episodes", res_dir)

    i2c.plot_alphas(res_dir, "final")
    i2c.plot_cost(res_dir, "cost_final")

    policy_linear.write(*i2c.get_local_linear_policy())
    x_final, y_final, _, _ = env.run(policy_linear)
    s_est = i2c.get_marginal_trajectory()
    env.plot_sim(x_final, s_est, "Final", res_dir)
    # generate gif for mujoco envs
    env.run_render(policy_linear, res_dir)

    policy_linear.zero()
    policy_linear.k = i2c.get_marginal_input().reshape(policy_linear.k.shape)
    x_ff, _, _, _ = env.run(policy_linear)
    env.plot_sim(x_ff, s_est, "Final Feedforward", res_dir)

    # save model and data
    save_trajectories(x_final, y_final, i2c, res_dir)
    traj_eval.save("episodic", res_dir)
    traj_eval_iter.save("iter", res_dir)
    i2c.save(res_dir, f"{i}")

    i2c.close()
    env.close()


def save_trajectories(s, dx, inference, res_dir):
    if res_dir:
        x_real = s[:, : inference.sys.dim_x]
        u_real = s[:, inference.sys.dim_x :]
        np.save(os.path.join(res_dir, "xu_real.npy"), s)
        np.save(os.path.join(res_dir, "dx_real.npy"), dx)
        np.save(os.path.join(res_dir, "x_real.npy"), x_real)
        np.save(os.path.join(res_dir, "u_real.npy"), u_real)
        inference.save_traj(res_dir)


if __name__ == "__main__":
    import argparse
    from glob import glob
    import importlib
    from os.path import basename, join, splitext
    from i2c.utils import write_commit

    DIR_NAME = os.path.dirname(__file__)

    configure_plots()

    exps = [
        splitext(basename(e))[0]
        for e in glob(os.path.join(DIR_NAME, "experiments", "*.py"))
        if "__init__" not in e
    ]

    parser = argparse.ArgumentParser(description="Run Experiment")
    parser.add_argument("config", help="file with hyperparameters", choices=exps)
    parser.add_argument("-n", "--name", help="folder", default="")
    parser.add_argument("-w", "--weights", help="path to model weights")
    parser.add_argument("-r", "--random-seed", help="random seed", default=0)
    parser.add_argument(
        "--release", action="store_true", help="Final reported version, no timestamp"
    )

    args = parser.parse_args()

    set_seed(args.random_seed)

    experiment = importlib.import_module(f"experiments.{args.config}")
    exp_filename = f"{args.config}.py"
    exp_file = join(DIR_NAME, "experiments", exp_filename)
    res_dir = make_results_folder(
        f"i2c_{args.config}", args.random_seed, args.name, release=args.release
    )
    # copy experiment config and git commit to results
    copyfile(exp_file, os.path.join(res_dir, exp_filename))
    write_commit(res_dir)

    setup_logger(res_dir)
    try:
        run(experiment, res_dir, args.weights)
    except:
        logging.exception("Experiment failed:")
        raise
    plt.show()
