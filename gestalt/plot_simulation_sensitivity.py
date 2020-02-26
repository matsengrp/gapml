"""
Plot out how error rates might affect method accuracy
"""

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
            description='plot tree fitting method results wrt error rates')
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
        type=str)
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default="_output/scratch")
    parser.add_argument(
        '--errors-list',
        type=str,
        default="0,0.1")
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
    args.errors_list = parse_comma_str(args.errors_list, float)
    return args

def get_true_model(
        args,
        seed,
        n_bcodes,
        error,
        measurer_classes=[BHVDistanceMeasurer]):
    print(args.true_model_file_template)
    file_name = args.true_model_file_template % (error, seed)
    model_params, assessor = file_readers.read_true_model(
            file_name,
            n_bcodes,
            measurer_classes=measurer_classes,
            scratch_dir=args.scratch_dir,
            use_error_prone_alleles=True)
    return model_params, assessor

def get_mle_result(args, seed, n_bcodes, error):
    file_name = args.mle_file_template % (error, seed, n_bcodes)
    print(file_name)
    try:
        with open(file_name, "rb") as f:
            mle_model = six.moves.cPickle.load(f)["final_fit"]
    except Exception:
        raise FileNotFoundError("nope %s" % file_name)
    return mle_model.model_params_dict, mle_model.fitted_bifurc_tree

def main(args=sys.argv[1:]):
    args = parse_args(args)
    perf_key = "full_bhv"

    all_perfs = []
    for error in args.errors_list:
        for n_bcodes in args.n_bcodes_list:
            print("barcodes", n_bcodes)
            for seed in args.data_seeds:
                print("seed", seed)
                true_params, assessor = get_true_model(args, seed, n_bcodes, error)

                try:
                    print("mle")
                    mle_params, mle_tree = get_mle_result(args, seed, n_bcodes, error)

                    mle_perf_dict = assessor.assess(mle_tree, mle_params)
                    res_dict = {
                        'Error rate': error,
                        'barcodes': n_bcodes,
                        'seed': seed,
                        'Performance metric': perf_key,
                        'Value': mle_perf_dict[perf_key]}
                    all_perfs.append(res_dict)
                except FileNotFoundError:
                    print("not found mle", n_bcodes, seed)


    all_perfs = pd.DataFrame(all_perfs)
    print(all_perfs)
    sns.set_context("paper", font_scale = 1.8)
    sns_plot = sns.relplot(
            x="barcodes",
            y="Value",
            hue="Error rate",
            kind="line",
            data=all_perfs,
    )
    #sns_plot.fig.get_axes()[0].set_yscale('log')
    sns_plot.savefig(args.out_plot)


if __name__ == "__main__":
    main()
