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
        measurer_classes: list):
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
    try:
        with open(file_name, "rb") as f:
            mle_model = six.moves.cPickle.load(f)["final_fit"]
    except Exception:
        raise FileNotFoundError("nope %s" % file_name)
    return mle_model.model_params_dict, mle_model.fitted_bifurc_tree

def main(args=sys.argv[1:]):
    args = parse_args(args)
    perf_keys = {
            "full_bhv": BHVDistanceMeasurer,
            "full_internal_pearson": InternalCorrMeasurer}

    all_perfs = []
    for error in args.errors_list:
        for n_bcodes in args.n_bcodes_list:
            for seed in args.data_seeds:
                try:
                    true_params, assessor = get_true_model(args, seed, n_bcodes, error, measurer_classes=list(perf_keys.values()))
                    mle_params, mle_tree = get_mle_result(args, seed, n_bcodes, error)

                    mle_perf_dict = assessor.assess(mle_tree, mle_params)
                    for k, v in perf_keys.items():
                        res_dict = {
                            'Error rate': error,
                            'barcodes': n_bcodes,
                            'seed': seed,
                            'Tree error measure': k,
                            'Value': mle_perf_dict[k]}
                        all_perfs.append(res_dict)
                except FileNotFoundError:
                    print("not found mle", n_bcodes, seed)


    all_perfs = pd.DataFrame(all_perfs)
    print(all_perfs)
    sns.set_context("paper", font_scale = 1.8)
    if len(args.n_bcodes_list) == 1:
        if len(perf_keys) == 1:
            sns_plot = sns.relplot(
                x="Error rate",
                y="Value",
                kind="line",
                data=all_perfs,
            )
        else:
            sns_plot = sns.relplot(
                x="Error rate",
                y="Value",
                col="Tree error measure",
                kind="line",
                data=all_perfs,
                facet_kws={"sharey": False},
            )
    else:
        sns_plot = sns.relplot(
            x="barcodes",
            y="Value",
            hue="Error rate",
            kind="line",
            data=all_perfs,
        )
    sns_plot.savefig(args.out_plot)


if __name__ == "__main__":
    main()
