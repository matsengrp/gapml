import numpy as np
import sys
import argparse
import os
import six
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from tree_distance import BHVDistanceMeasurer, InternalCorrMeasurer
from plot_simulation_topol_consist import get_true_model
from common import parse_comma_str
import file_readers


def parse_args(args):
    parser = argparse.ArgumentParser(
            description='plot the training history')
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
        default=",".join(map(str, range(300,301))))
    parser.add_argument(
        '--pmle-plot',
        type=str,
        default="_output/%s_training_hist_grad_plot_%d_%d.png")
    parser.add_argument(
        '--chad-plot',
        type=str,
        default="_output/%s_training_hist_chad_plot_%d_%d.png")
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
            mle_model_hist = six.moves.cPickle.load(f)["tuning_history"]
    except Exception:
        raise FileNotFoundError("nope %s" % file_name)
    return mle_model_hist

def plot_fixed_chad_training_history(result, out_file):
    dist_key = "full_bhv"
    Y_bhv = []
    X_iters = []

    train_hist = result[0]['best_res'].train_history
    for train_iter in train_hist:
        if 'performance' in train_iter:
            Y_bhv.append(train_iter['performance'][dist_key])
            X_iters.append(train_iter['iter'])
    df = pd.DataFrame({
        "Iteration": X_iters,
        "BHV": Y_bhv})
    print(df)
    plt.clf()
    sns.lineplot(x="Iteration", y="BHV", data=df)
    plt.savefig(out_file)
    print(out_file)

def plot_chad_training_history(result, out_file):
    dist_key = "full_bhv"
    Y_bhv = []
    X_iters = []

    train_iter_res = result[0]['best_res'].train_history[0]['performance']
    Y_bhv.append(train_iter_res[dist_key])
    X_iters.append(0)
    for idx, train_iter_res in enumerate(result):
        print(train_iter_res.keys())
        train_iter_res = train_iter_res['best_res'].train_history[-1]['performance']
        Y_bhv.append(train_iter_res[dist_key])
        X_iters.append(idx + 1)
    df = pd.DataFrame({
        "Iteration": X_iters,
        "BHV": Y_bhv})
    print(df)
    plt.clf()
    sns.lineplot(x="Iteration", y="BHV", data=df)
    plt.savefig(out_file)
    print(out_file)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    sns.set_context("paper", font_scale=1.5)
    for n_bcodes in args.n_bcodes_list:
        for seed in args.data_seeds:
            #true_params, assessor = get_true_model(args, seed, n_bcodes)
            mle_hist = get_mle_result(args, seed, n_bcodes)
            plot_fixed_chad_training_history(
                    mle_hist,
                    args.pmle_plot % (args.simulation_folder, seed, n_bcodes))
            plot_chad_training_history(
                    mle_hist,
                    args.chad_plot % (args.simulation_folder, seed, n_bcodes))


if __name__ == "__main__":
    main()
