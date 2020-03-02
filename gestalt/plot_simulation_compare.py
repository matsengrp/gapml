# Just get the CI for the error for different metrics
import numpy as np
import sys
import argparse
import os
import six
import pandas as pd

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
        '--chronos-file-template',
        type=str)
    parser.add_argument(
        '--nj-file-template',
        type=str)
    parser.add_argument(
        '--data-seeds',
        type=str,
        default=",".join(map(str, range(20,40))))
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default="_output/scratch")

    parser.set_defaults(n_bcodes_list=[1])
    args = parser.parse_args(args)

    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    args.data_seeds = parse_comma_str(args.data_seeds, int)
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
            measurer_classes=[BHVDistanceMeasurer, InternalCorrMeasurer, UnrootRFDistanceMeasurer])
    return model_params, assessor

def get_mle_result(args, seed, n_bcodes):
    file_name = args.mle_file_template % (seed, n_bcodes)
    try:
        with open(file_name, "rb") as f:
            mle_model = six.moves.cPickle.load(f)["final_fit"]
    except Exception:
        raise FileNotFoundError("nope %s" % file_name)
    return mle_model.model_params_dict, mle_model.fitted_bifurc_tree

def get_neighbor_joining_result(args, seed, n_bcodes, assessor, perf_measure="full_bhv"):
    file_name = args.nj_file_template % (
                seed,
                n_bcodes)
    with open(file_name, "rb") as f:
        fitted_models = six.moves.cPickle.load(f)

    perfs = []
    for neighbor_joining_model in fitted_models:
        perf_dict = assessor.assess(neighbor_joining_model["fitted_tree"])
        perfs.append(perf_dict[perf_measure])
    best_perf_idx = np.argmin(perfs)
    return None, fitted_models[best_perf_idx]["fitted_tree"]

def get_chronos_result(args, seed, n_bcodes, assessor, perf_measure="full_bhv"):
    file_name = args.chronos_file_template % (
                seed,
                n_bcodes)
    with open(file_name, "rb") as f:
        fitted_models = six.moves.cPickle.load(f)

    perfs = []
    for chronos_model in fitted_models:
        perf_dict = assessor.assess(chronos_model["fitted_tree"])
        perfs.append(perf_dict[perf_measure])
    best_perf_idx = np.argmin(perfs)
    return None, fitted_models[best_perf_idx]["fitted_tree"]

def main(args=sys.argv[1:]):
    args = parse_args(args)
    all_perfs = []
    for n_bcodes in args.n_bcodes_list:
        for seed in args.data_seeds:
            true_params, assessor = get_true_model(args, seed, n_bcodes)

            try:
                mle_params, mle_tree = get_mle_result(args, seed, n_bcodes)
                mle_perf_dict = assessor.assess(mle_tree, mle_params)
                res_dict = {"method": "MLE"}
                for k in ["full_bhv", "full_internal_pearson"]:
                    res_dict[k] = mle_perf_dict[k]
                all_perfs.append(res_dict)
            except FileNotFoundError:
                print("not found mle", n_bcodes, seed)

            try:
                print("chronos")
                _, chronos_tree = get_chronos_result(
                        args,
                        seed,
                        n_bcodes,
                        assessor)
                chronos_perf_dict = assessor.assess(chronos_tree)
                res_dict = {"method": "chronos"}
                for k in ["full_bhv", "full_internal_pearson"]:
                    res_dict[k] = chronos_perf_dict[k]
                all_perfs.append(res_dict)
            except FileNotFoundError:
                print("not found chronos", n_bcodes, seed)

            try:
                print("nj")
                _, nj_tree = get_neighbor_joining_result(
                        args,
                        seed,
                        n_bcodes,
                        assessor)
                nj_perf_dict = assessor.assess(nj_tree)
                res_dict = {"method": "nj"}
                for k in ["full_bhv", "full_internal_pearson"]:
                    res_dict[k] = nj_perf_dict[k]
                all_perfs.append(res_dict)
            except FileNotFoundError:
                print("not found nj", n_bcodes, seed)

    # Plot MSE and BHV
    method_perfs = pd.DataFrame(all_perfs)
    print(method_perfs)
    mean = method_perfs.groupby('method').mean()
    std_err = np.sqrt(method_perfs.groupby('method').var()/method_perfs.shape[0])

    print("MEAN", mean)
    print("LOWER 95 CI", mean - 1.96 * std_err)
    print("LOWER 95 CI", mean + 1.96 * std_err)


if __name__ == "__main__":
    main()
