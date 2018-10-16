import sys
import six
import argparse
import os.path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import seaborn as sns
from common import parse_comma_str

def parse_args(args):
    parser = argparse.ArgumentParser(
            description='plot abundance vs internal node time')
    parser.add_argument(
        '--obs-file-template',
        type=str,
        default="_output/%s/fish_data_restrict.pkl")
    parser.add_argument(
        '--mle-file-template',
        type=str,
        default="_output/%s/sum_states_10/extra_steps_0/tune_pen.pkl")
    parser.add_argument(
        '--nj-file-template',
        type=str,
        default="_output/%s/nj_fitted.pkl")
    parser.add_argument(
        '--chronos-file-template',
        type=str,
        default="_output/%s/chronos_fitted.pkl")
    parser.add_argument(
        '--folder',
        type=str,
        default="analyze_gestalt")
    parser.add_argument(
        '--fishies',
        type=str,
        default="dome1_abund1")
        #default="dome1_abund1,dome3_abund1,dome5_abund1,dome8_abund1")
    parser.add_argument(
        '--out-plot-template',
        type=str,
        default="_output/time_to_abund_%s.png")
    parser.add_argument(
        '--tot-time',
        type=float,
        default="num hpf")
    args = parser.parse_args(args)
    args.fishies = parse_comma_str(args.fishies, str)
    return args

def load_data(args, fish, method):
    if method == "PMLE":
        fitted_tree_file = os.path.join(args.folder, args.mle_file_template % fish)
        with open(fitted_tree_file, "rb") as f:
            fitted_bifurc_tree = six.moves.cPickle.load(f)["final_fit"].fitted_bifurc_tree
    elif method == "chronos":
        chronos_tree_file = os.path.join(args.folder, args.chronos_file_template % fish)
        with open(chronos_tree_file, "rb") as f:
            fitted_bifurc_tree = six.moves.cPickle.load(f)[0]["fitted_tree"]
    elif method == "nj":
        nj_tree_file = os.path.join(args.folder, args.nj_file_template % fish)
        with open(nj_tree_file, "rb") as f:
            fitted_bifurc_tree = six.moves.cPickle.load(f)[1]["fitted_tree"]
    else:
        raise ValueError("nope method dont exist")

    obs_file = os.path.join(args.folder, args.obs_file_template % fish)
    with open(obs_file, "rb") as f:
        obs_dict = six.moves.cPickle.load(f)
    return fitted_bifurc_tree, obs_dict

def plot_distance_to_abundance(
        fitted_bifurc_tree,
        tot_time,
        out_plot_file="/Users/jeanfeng/Desktop/scatter_dist_to_abundance.png"):
    """
    Understand if distance to root is inversely related to total abundance
    """
    X_dists = []
    Y_abundance = []
    for node in fitted_bifurc_tree.traverse("preorder"):
        if node.is_leaf():
            continue
        if min([leaf.abundance for leaf in node]) > 1:
            continue
        dist = node.get_distance(fitted_bifurc_tree)
        X_dists.append(dist)
        tot_abundance = float(sum([leaf.abundance for leaf in node]))
        Y_abundance.append(tot_abundance)

    if out_plot_file:
        print(out_plot_file)
        pyplot.clf()
        sns.regplot(
                np.array(X_dists),
                np.log2(Y_abundance) - np.log2(np.max(Y_abundance)),
                robust=True)
        pyplot.xlabel("dist to root")
        pyplot.ylabel("log_2(abundance/max_abundance)")
        pyplot.xlim(-0.05,1)
        pyplot.savefig(out_plot_file)
    slope, _, _, pval, se = stats.linregress(X_dists, np.log2(Y_abundance))
    print("estimated slope", slope, "pval", pval)
    print("95 CI", slope - 1.96 * se, slope + 1.96 * se)
    print("estimated time btw divisions", -tot_time/slope * 60)
    print("95 CI", -tot_time * 60/(slope - 1.96 * se), -tot_time * 60/ (slope + 1.96 * se))


def main(args=sys.argv[1:]):
    args = parse_args(args)
    methods = ["PMLE", "chronos", "nj"]
    for method in methods:
        print("method", method)
        for fish in args.fishies:
            print(fish)
            fitted_bifurc_tree, obs_data_dict = load_data(args, fish, method)
            plot_distance_to_abundance(
                fitted_bifurc_tree,
                args.tot_time,
                out_plot_file = args.out_plot_template % fish)

if __name__ == "__main__":
    main()
