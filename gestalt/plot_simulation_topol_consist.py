import numpy as np
import sys
import argparse
import os
import six
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from tree_distance import BHVDistanceMeasurer, InternalCorrMeasurer
from common import parse_comma_str
import file_readers


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='tune over topologies and fit model parameters')
    parser.add_argument(
        '--true-model-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/true_model.pkl")
    parser.add_argument(
        '--mle-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/num_barcodes%d/sum_states_10/extra_steps_1/tune_fitted.pkl")
    parser.add_argument(
        '--chronos-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/num_barcodes%d/chronos_fitted.pkl")
    parser.add_argument(
        '--nj-file-template',
        type=str,
        default="_output/model_seed%d/%d/%s/num_barcodes%d/nj_fitted.pkl")
    parser.add_argument(
        '--simulation-folder',
        type=str,
        default="simulation_topol_consist")
    parser.add_argument(
        '--model-seed',
        type=int,
        default=100)
    parser.add_argument(
        '--data-seeds',
        type=str,
        default=",".join(map(str, range(400,410))))
    parser.add_argument(
        '--n-bcodes-list',
        type=str,
        default="1,2,4")
    parser.add_argument(
        '--growth-stage',
        type=str,
        default="small")
    parser.add_argument(
        '--out-plot',
        type=str,
        default="_output/simulation_topol_consist_plot.png")
    parser.add_argument(
        '--scratch-dir',
        type=str,
        default="_output/scratch")

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
        n_bcodes,
        measurer_classes=[BHVDistanceMeasurer, InternalCorrMeasurer]):
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
    plot_perf_measures = [
            "full_bhv",
            "collapse_bhv",
            "full_internal_pearson",
            "collapse_internal_pearson"]
    plot_perf_measures_set = set(plot_perf_measures)

    all_perfs = []
    for n_bcodes in args.n_bcodes_list:
        print("barcodes", n_bcodes)
        for seed in args.data_seeds:
            print("seed", seed)
            true_params, assessor = get_true_model(args, seed, n_bcodes)

            try:
                mle_params, mle_tree = get_mle_result(args, seed, n_bcodes)
                mle_perf_dict = assessor.assess(mle_tree, mle_params)
                for k, v in mle_perf_dict.items():
                    if k in plot_perf_measures_set:
                        small_dict = {
                                'method': 'mle',
                                'n_bcodes': n_bcodes,
                                'seed': seed,
                                'perf_meas': k,
                                'value': v}
                        all_perfs.append(small_dict)
            except FileNotFoundError:
                print("not found mle", n_bcodes, seed)

            try:
                _, chronos_tree = get_chronos_result(args, seed, n_bcodes, assessor)
                chronos_perf_dict = assessor.assess(chronos_tree)
                for k, v in chronos_perf_dict.items():
                    if k in plot_perf_measures_set:
                        small_dict = {
                                'method': 'chronos',
                                'n_bcodes': n_bcodes,
                                'seed': seed,
                                'perf_meas': k,
                                'value': v}
                        all_perfs.append(small_dict)
            except FileNotFoundError:
                print("not found chronos", n_bcodes, seed)

            try:
                _, nj_tree = get_neighbor_joining_result(args, seed, n_bcodes, assessor)
                nj_perf_dict = assessor.assess(nj_tree)
                for k, v in nj_perf_dict.items():
                    if k in plot_perf_measures_set:
                        small_dict = {
                                'method': 'nj',
                                'n_bcodes': n_bcodes,
                                'seed': seed,
                                'perf_meas': k,
                                'value': v}
                        all_perfs.append(small_dict)
            except FileNotFoundError:
                print("not found nj", n_bcodes, seed)

    method_perfs = pd.DataFrame(all_perfs)
    print(method_perfs)
    sns_plot = sns.catplot(
            x="n_bcodes",
            y="value",
            hue="method",
            col="perf_meas",
            data=method_perfs,
            kind="point",
            col_wrap=2,
            col_order=plot_perf_measures,
            sharey=False)
    sns_plot.savefig(args.out_plot)


if __name__ == "__main__":
    main()

#ONE_TREE_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/tune_example_refitnew_tree0.pkl" % (prefix, model_seed, seeds[0], 3)
#ONE_TREE_PLOT_TEMPLATE = "%ssimulation_topol_consist/_output/model_seed%d/%d/lambda_diff/num_barcodes%d/tune_example_refitnew_tree0.png" % (prefix, model_seed, seeds[0], 3)
#with open(ONE_TREE_TEMPLATE, "rb") as f:
#    result = six.moves.cPickle.load(f)
#
#dist_key = "bhv"
#Y_bhv = []
#Y_pen_log_lik = []
#X_iters = []
#for train_iter_res in result["raw"].train_history:
#    if 'tree_dists' in train_iter_res:
#        Y_bhv.append(train_iter_res['tree_dists'][dist_key])
#        Y_pen_log_lik.append(train_iter_res['pen_log_lik'])
#        X_iters.append(train_iter_res['iter'])
#last_raw_iter = X_iters[-1]
#for train_iter_res in result["refit"].train_history:
#    if 'tree_dists' in train_iter_res:
#        Y_bhv.append(train_iter_res['tree_dists'][dist_key])
#        Y_pen_log_lik.append(train_iter_res['pen_log_lik'])
#        X_iters.append(last_raw_iter + train_iter_res['iter'])
#
#plt.clf()
#plt.figure(1)
#plt.subplot(211)
#plt.plot(X_iters, Y_bhv)
#plt.ylabel("%s distance" % dist_key)
#plt.subplot(212)
#plt.plot(X_iters, Y_pen_log_lik)
#plt.ylabel("pen log lik")
#plt.xlabel("Iterations")
#plt.savefig(ONE_TREE_PLOT_TEMPLATE)
#print(ONE_TREE_PLOT_TEMPLATE)
