import numpy as np
import sys
import argparse
import os
import six
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from tree_distance import BHVDistanceMeasurer, InternalCorrMeasurer, UnrootRFDistanceMeasurer
from common import parse_comma_str
import file_readers

def parse_args(args):
    parser = argparse.ArgumentParser(
            description='plot tree fitting method results wrt number of barcodes. evaluate mle consistency, bias, variance')
    parser.add_argument(
        '--true-model-file-template',
        type=str)
    parser.add_argument(
        '--mle-file-template',
        type=str)
    parser.add_argument(
        '--data-seeds',
        type=str,
        default=",".join(map(str, range(20,40))))
    parser.add_argument(
        '--out-plot',
        type=str,
        default="_output/%s_plot_test.png")
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default="_output/scratch")
    parser.add_argument(
        '--n-bcodes-list',
        type=str,
        default="1,2,4")

    parser.set_defaults()
    args = parser.parse_args(args)

    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    args.data_seeds = parse_comma_str(args.data_seeds, int)
    args.n_bcodes_list = parse_comma_str(args.n_bcodes_list, int)
    return args

def get_true_model(
        args,
        seed,
        n_bcodes):
    file_name = args.true_model_file_template % seed
    model_params, assessor = file_readers.read_true_model(
            file_name,
            n_bcodes,
            scratch_dir=args.scratch_dir,
            measurer_classes=[BHVDistanceMeasurer])
    return model_params, assessor

def get_mle_result(args, seed, n_bcodes):
    file_name = args.mle_file_template % (seed, n_bcodes)
    try:
        with open(file_name, "rb") as f:
            mle_model = six.moves.cPickle.load(f)["final_fit"]
    except Exception:
        raise FileNotFoundError("nope %s" % file_name)
    return mle_model.model_params_dict, mle_model.fitted_bifurc_tree

def main(args=sys.argv[1:]):
    args = parse_args(args)
    all_perfs = []
    all_estimates = {n_bcodes: [] for n_bcodes in args.n_bcodes_list}
    all_true_vals = {n_bcodes: [] for n_bcodes in args.n_bcodes_list}
    for n_bcodes in args.n_bcodes_list:
        for seed in args.data_seeds:
            true_params, assessor = get_true_model(args, seed, n_bcodes)

            try:
                mle_params, mle_tree = get_mle_result(args, seed, n_bcodes)
                mle_perf_dict = assessor.assess(mle_tree, mle_params)
                mle_estimates = []
                true_vals = []
                for k, mle_estimate in mle_params.items():
                    if k in ["tot_time", "tot_time_extra", "target_lam_decay_rate"]:
                        continue
                    true_val = np.array(true_params[k])
                    if true_val.size == 0:
                        continue
                    true_vals.append(true_val.flatten())
                    mle_estimates.append(mle_estimate.flatten())
                    assert true_val.size == mle_estimate.size

                true_vals = np.concatenate(true_vals)
                mle_estimates = np.concatenate(mle_estimates)
                all_perfs.append({
                    "Param": "Mutation parameters",
                    "Error": np.mean(np.power(mle_estimates - true_vals, 2)),
                    "Number of barcodes": n_bcodes,
                })
                all_perfs.append({
                    "Param": "Tree",
                    "Error": mle_perf_dict["full_bhv"],
                    "Number of barcodes": n_bcodes,
                })
                all_estimates[n_bcodes].append(mle_estimates)
                all_true_vals[n_bcodes].append(true_vals)
            except FileNotFoundError:
                print("not found mle", n_bcodes, seed)

    # Plot MSE and BHV
    method_perfs = pd.DataFrame(all_perfs)
    print(method_perfs)
    sns.set_context("paper", font_scale = 1.8)
    sns_plot = sns.relplot(
            x="Number of barcodes",
            y="Error",
            hue="Param",
            kind="line",
            data=method_perfs,
    )
    sns_plot.fig.get_axes()[0].set_yscale('log')
    sns_plot._legend.set_title("")
    sns_plot.savefig(args.out_plot)

    # Plot bias and variance now
    print(all_true_vals)
    biases = [np.mean(np.array(all_estimates[n_bcodes]) - np.array(all_true_vals[n_bcodes])) for n_bcodes in args.n_bcodes_list]
    variances = [np.mean(np.var(all_estimates[n_bcodes], axis=0)) for n_bcodes in args.n_bcodes_list]
    bias_var_all = pd.DataFrame({
        "Value": np.concatenate([biases, variances]),
        "Number of barcodes": np.concatenate([args.n_bcodes_list, args.n_bcodes_list]),
        "Measure": ["Bias"] * len(args.n_bcodes_list) + ["Variance"] * len(args.n_bcodes_list),
    })
    print(bias_var_all)
    sns_plot = sns.relplot(
        x="Number of barcodes",
        y="Value",
        col="Measure",
        data=bias_var_all,
        kind="scatter",
    )


if __name__ == "__main__":
    main()
