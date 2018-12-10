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
            description='plot tree fitting method results wrt number of barcodes')
    parser.add_argument(
        '--true-model-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/phantom0/true_model.pkl")
    parser.add_argument(
        '--mle-file-template',
        type=str,
        #default="_output/model_seed%d/%d/%s/phantom0/num_barcodes%d/sum_states_30/extra_steps_2/tune_fitted_new.pkl")
        default="_output/model_seed%d/%d/%s/phantom0/num_barcodes%d/sum_states_20/extra_steps_1/tune_fitted.pkl")
    parser.add_argument(
        '--chronos-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/phantom0/num_barcodes%d/chronos_fitted.pkl")
    parser.add_argument(
        '--nj-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/phantom0/num_barcodes%d/nj_fitted.pkl")
    parser.add_argument(
        '--simulation-folder',
        type=str,
        #default="simulation_topol_consist")
        default="simulation_compare")
    parser.add_argument(
        '--model-seed',
        type=int,
        #default=101)
        default=100)
    parser.add_argument(
        '--data-seeds',
        type=str,
        #default=",".join(map(str, range(20,40))))
        default=",".join(map(str, range(300,320))))
    parser.add_argument(
        '--out-plot',
        type=str,
        default="_output/%s_plot.png")
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default="_output/scratch")

    parser.set_defaults()
    args = parser.parse_args(args)

    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)

    args.data_seeds = parse_comma_str(args.data_seeds, int)
    args.n_bcodes_list = [1,2,4] if args.simulation_folder == "simulation_topol_consist" else [1]
    args.growth_stage = "small" if args.simulation_folder == "simulation_topol_consist" else "30hpf"
    return args

def get_true_model(
        args,
        seed,
        n_bcodes,
        measurer_classes=[BHVDistanceMeasurer, InternalCorrMeasurer, UnrootRFDistanceMeasurer]):
    file_name = os.path.join(args.simulation_folder,
            args.true_model_file_template % (
                args.model_seed,
                seed,
                args.growth_stage))
    model_params, assessor = file_readers.read_true_model(
            file_name,
            n_bcodes,
            measurer_classes=measurer_classes,
            scratch_dir=args.scratch_dir)
    return model_params, assessor

def get_mle_result(args, seed, n_bcodes):
    file_name = os.path.join(
            args.simulation_folder,
            args.mle_file_template % (
                args.model_seed,
                seed,
                args.growth_stage,
                n_bcodes))
    print(file_name)
    try:
        with open(file_name, "rb") as f:
            mle_model = six.moves.cPickle.load(f)["final_fit"]
    except Exception:
        raise FileNotFoundError("nope %s" % file_name)
    return mle_model.model_params_dict, mle_model.fitted_bifurc_tree

def get_neighbor_joining_result(args, seed, n_bcodes, assessor, perf_measure="full_bhv"):
    file_name = os.path.join(
            args.simulation_folder,
            args.nj_file_template % (
                args.model_seed,
                seed,
                args.growth_stage,
                n_bcodes))
    with open(file_name, "rb") as f:
        fitted_models = six.moves.cPickle.load(f)

    perfs = []
    for neighbor_joining_model in fitted_models:
        perf_dict = assessor.assess(neighbor_joining_model["fitted_tree"])
        perfs.append(perf_dict[perf_measure])
    best_perf_idx = np.argmin(perfs)
    return None, fitted_models[best_perf_idx]["fitted_tree"]

def get_chronos_result(args, seed, n_bcodes, assessor, perf_measure="full_bhv"):
    file_name = os.path.join(
            args.simulation_folder,
            args.chronos_file_template % (
                args.model_seed,
                seed,
                args.growth_stage,
                n_bcodes))
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
    LABEL_DICT = {
        "full_bhv": "BHV",
        "full_internal_pearson": "1 - Internal corr",
        "full_ete_rf_unroot": "RF",
    }
    plot_perf_measures = [
            "full_bhv",
            "full_internal_pearson",
            "full_ete_rf_unroot",
    ]
    plot_perf_measures_set = set(plot_perf_measures)

    all_perfs = []
    for n_bcodes in args.n_bcodes_list:
        print("barcodes", n_bcodes)
        for seed in args.data_seeds:
            print("seed", seed)
            true_params, assessor = get_true_model(args, seed, n_bcodes)

            try:
                print("mle")
                mle_params, mle_tree = get_mle_result(args, seed, n_bcodes)
                mle_perf_dict = assessor.assess(mle_tree, mle_params)
                for k, v in mle_perf_dict.items():
                    if k in plot_perf_measures_set:
                        small_dict = {
                                'Method': 'GapML',
                                'Number of barcodes': n_bcodes,
                                'seed': seed,
                                'Performance metric': LABEL_DICT[k],
                                'Value': v}
                        all_perfs.append(small_dict)
            except FileNotFoundError:
                print("not found mle", n_bcodes, seed)

            try:
                print("chronos")
                _, chronos_tree = get_chronos_result(
                        args,
                        seed,
                        n_bcodes,
                        assessor,
                        plot_perf_measures[0])
                chronos_perf_dict = assessor.assess(chronos_tree)
                for k, v in chronos_perf_dict.items():
                    if k in plot_perf_measures_set:
                        small_dict = {
                                'Method': 'CS + chronos',
                                'Number of barcodes': n_bcodes,
                                'seed': seed,
                                'Performance metric': LABEL_DICT[k],
                                'Value': v}
                        all_perfs.append(small_dict)
            except FileNotFoundError:
                print("not found chronos", n_bcodes, seed)

            try:
                print("nj")
                _, nj_tree = get_neighbor_joining_result(
                        args,
                        seed,
                        n_bcodes,
                        assessor,
                        plot_perf_measures[0])
                nj_perf_dict = assessor.assess(nj_tree)
                for k, v in nj_perf_dict.items():
                    if k in plot_perf_measures_set:
                        small_dict = {
                                'Method': 'NJ + chronos',
                                'Number of barcodes': n_bcodes,
                                'seed': seed,
                                'Performance metric': LABEL_DICT[k],
                                'Value': v}
                        all_perfs.append(small_dict)
            except FileNotFoundError:
                print("not found nj", n_bcodes, seed)

    method_perfs = pd.DataFrame(all_perfs)
    print(method_perfs)
    sns.set_context("paper", font_scale = 1.8)
    sns_plot = sns.catplot(
            x="Number of barcodes",
            y="Value",
            hue="Method",
            col="Performance metric",
            data=method_perfs,
            kind="point",
            col_wrap=1,
            col_order=[LABEL_DICT[k] for k in plot_perf_measures],
            sharey=False)
    sns_plot.set_titles("{col_name}")
    sns_plot.savefig(args.out_plot % args.simulation_folder)


if __name__ == "__main__":
    main()
