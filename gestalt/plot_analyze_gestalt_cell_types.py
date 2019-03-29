import sys
import random
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import seaborn as sns
import numpy as np
import scipy.stats
import pandas as pd
from scipy import stats

from common import assign_rand_tree_lengths
from plot_analyze_gestalt_meta import load_fish, get_allele_to_cell_states, create_shuffled_cell_state_abund_labels

ECTO = 0
NEURAL_CREST = 1
ENDO = 2
MESO = 3
ORGAN_GERM_LAYERS = {
    "Brain": ECTO,
    "Eye1": ECTO,
    "Eye2": ECTO,
    "Gills": NEURAL_CREST,
    "Intestine": ENDO,
    "Upper_GI": ENDO,
    "Blood": MESO,
    "Heart_GFP-": MESO,
    "Heart_diss": MESO,
    "Heart_GFP+": MESO,
    "Heart_chunk": MESO,
}

def parse_args(args):
    parser = argparse.ArgumentParser(
            description='plot time in tree vs number of descendant cell types/germ layers')
    parser.add_argument(
        '--num-rands',
        type=int,
        default=2000)
    parser.add_argument(
        '--obs-file',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/fish_data_restrict.pkl")
    parser.add_argument(
        '--mle-template',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/sum_states_20/extra_steps_1/tune_pen_hanging.pkl")
    parser.add_argument(
        '--chronos-template',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/chronos_fitted.pkl")
    parser.add_argument(
        '--nj-template',
        type=str,
        default="analyze_gestalt/_output/%s/sampling_seed0/nj_fitted.pkl")
    parser.add_argument(
        '--null-method',
        type=str,
        default=None,
        help="If none, use the method itself as reference. Otherwise always use this method as reference.")
    parser.add_argument(
        '--out-germ-layer-plot-template',
        type=str,
        default="_output/time_to_germ_layer_%s_%s.png")
    parser.add_argument(
        '--out-cell-type-plot-template',
        type=str,
        default="_output/time_to_cell_type_%s_%s.png")
    args = parser.parse_args(args)
    return args

def get_distance_to_num_germ_layers(
        fitted_bifurc_tree,
        allele_to_cell_state,
        organ_dict):
    """
    """
    X_dists = []
    Y_n_germ_layers = []
    for node in fitted_bifurc_tree.traverse('postorder'):
        if node.is_leaf():
            allele_str = node.allele_events_list_str
            tissue_types = list(allele_to_cell_state[allele_str].keys())
            node.add_feature("germ_layers", set([
                ORGAN_GERM_LAYERS[organ_dict[c_state].replace("7B_", "")] for c_state in tissue_types]))
        else:
            node.add_feature("germ_layers", set())
            for child in node.children:
                node.germ_layers.update(child.germ_layers)

    for node in fitted_bifurc_tree.traverse('postorder'):
        if node.is_leaf():
            continue
        #if len(node.germ_layers) == 4:
        #    continue
        X_dists.append(node.dist_to_root)
        n_germ_layers = len(node.germ_layers)
        Y_n_germ_layers.append(n_germ_layers)
    return X_dists, Y_n_germ_layers

def plot_distance_to_num_germ_layers(
        X_dists,
        Y_n_germ_layers,
        out_plot_file):
    print(out_plot_file)
    pyplot.clf()
    data = pd.DataFrame.from_dict({
        "x":np.array(X_dists),
        "y":np.array(Y_n_germ_layers)})
    sns.boxplot(
            x="x", y="y", data=data,
            order=[4,3,2,1],
            orient='h',
            linewidth=2.5)
    sns.swarmplot(x="x", y="y", data=data,
            order=[4,3,2,1],
            orient='h',
            color=".25")
    pyplot.xlim(0,1)
    pyplot.ylabel("Number of descendant germ layers")
    pyplot.xlabel("Distance from root")
    pyplot.tight_layout()
    pyplot.savefig(out_plot_file)

def get_distance_to_num_cell_states(
        fitted_bifurc_tree,
        allele_to_cell_state,
        organ_dict):
    for node in fitted_bifurc_tree.traverse('postorder'):
        if node.is_leaf():
            allele_str = node.allele_events_list_str
            node.add_feature("cell_types", set(list(allele_to_cell_state[allele_str].keys())))
        else:
            node.add_feature("cell_types", set())
            for child in node.children:
                node.cell_types.update(child.cell_types)

    X_dists = []
    Y_n_cell_states = []
    for node in fitted_bifurc_tree.traverse('postorder'):
        if node.is_leaf():
            continue
        #if len(node.cell_types) == 11:
        #    continue
        X_dists.append(node.dist_to_root)
        Y_n_cell_states.append(len(node.cell_types))
    return X_dists, Y_n_cell_states

def plot_distance_to_num_cell_states(
        X_dists,
        Y_n_cell_states,
        out_plot_file):
    """
    Understand if distance to root is inversely related to number of
    different cell states in leaves
    """
    print(out_plot_file)
    pyplot.clf()
    data = pd.DataFrame.from_dict({
        "x":np.array(X_dists),
        "y":np.array(Y_n_cell_states)})
    sns.boxplot(
            x="x", y="y", data=data,
            order=np.arange(11,0,-1),
            orient='h',
            linewidth=2.5)
    sns.swarmplot(x="x", y="y", data=data,
            order=np.arange(11,0,-1),
            orient='h',
            color=".25")
    pyplot.xlim(0,1)
    pyplot.ylabel("Number of descendant cell types")
    pyplot.xlabel("Distance from root")
    pyplot.tight_layout()
    pyplot.savefig(out_plot_file)


def plot_branch_len_time(
    fitted_bifurc_tree,
    out_plot_file):
    """
    Plot fitted time vs branch length
    """
    X_dist = []
    Y_branch_len = []
    leaf_branch_lens = []
    for node in fitted_bifurc_tree.get_descendants():
        if not node.is_leaf():
            node_dist = node.get_distance(fitted_bifurc_tree)
            X_dist.append(node_dist)
            Y_branch_len.append(node.dist)
        else:
            leaf_branch_lens.append(node.dist)

    if out_plot_file:
        print(out_plot_file)
        pyplot.clf()
        sns.regplot(
                np.array(X_dist),
                np.array(Y_branch_len),
                lowess=True)
        pyplot.savefig(out_plot_file)
    print("branch len vs node time", stats.linregress(X_dist, Y_branch_len))
    print("leaf branch lens mean", np.mean(leaf_branch_lens), np.min(leaf_branch_lens), np.max(leaf_branch_lens))

def do_hypothesis_test(
        estimated_X_Y,
        null_tree,
        allele_to_cell_state,
        organ_dict,
        get_X_Y_func,
        num_rands):
    """
    Formally, the null hypothesis is the cell types and abundances are completely
    randomly assigned to the leaves of the two trees.
    """
    slope, _, corr, stupid_pval, stderr = scipy.stats.linregress(estimated_X_Y[1], estimated_X_Y[0])
    all_rand_slopes = []
    all_rand_corrs = []
    for _ in range(num_rands):
        shuffled_meta = create_shuffled_cell_state_abund_labels(allele_to_cell_state)
        rand_X, rand_Y = get_X_Y_func(null_tree, shuffled_meta, organ_dict)
        rand_slope, _, rand_corr, _, _ = scipy.stats.linregress(rand_Y, rand_X)
        all_rand_corrs.append(rand_corr)
        all_rand_slopes.append(rand_slope)
    print("corr %.03f" % corr)
    print("random mean corr %.03f" % np.mean(all_rand_corrs))
    print('p-val %.03f' % (np.mean(np.abs(corr) < np.abs(all_rand_corrs))))
    #print("slope %.03f (%.03f, %.03f)" % (slope, slope - 1.96 * stderr, slope + 1.96 * stderr))
    #print("random mean slope %.03f" % np.mean(all_rand_slopes))
    #print('p-val', np.mean(np.abs(slope) < np.abs(all_rand_slopes)))

def main(args=sys.argv[1:]):
    args = parse_args(args)
    random.seed(1)
    np.random.seed(1)
    fishies = ["ADR1", "ADR2"]
    methods = ["PMLE", "chronos", "nj"]
    sns.set_context("paper", font_scale=1.5)
    for method in methods:
        print("METHOD", method)
        null_method = method if args.null_method is None else args.null_method
        for fish in fishies:
            print("FISH", fish)
            null_tree, _ = load_fish(fish, args, null_method)
            assign_rand_tree_lengths(null_tree, 1)
            null_tree.label_dist_to_roots()

            tree, obs_dict = load_fish(fish, args, method)
            tree.label_dist_to_roots()

            allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)
            organ_dict = obs_dict["organ_dict"]

            print("GERM LAYERS")
            X_dists, Y_n_germ_layers = get_distance_to_num_germ_layers(
                tree,
                allele_to_cell_state,
                organ_dict)
            do_hypothesis_test(
                (X_dists, Y_n_germ_layers),
                null_tree,
                allele_to_cell_state,
                organ_dict,
                get_distance_to_num_germ_layers,
                args.num_rands)
            plot_distance_to_num_germ_layers(
                X_dists,
                Y_n_germ_layers,
                args.out_germ_layer_plot_template % (fish, method))

            print("TISSUE TYPES")
            X_dists, Y_n_cell_states = get_distance_to_num_cell_states(
                tree,
                allele_to_cell_state,
                organ_dict)
            do_hypothesis_test(
                (X_dists, Y_n_cell_states),
                null_tree,
                allele_to_cell_state,
                organ_dict,
                get_distance_to_num_cell_states,
                args.num_rands)
            plot_distance_to_num_cell_states(
                X_dists,
                Y_n_cell_states,
                args.out_cell_type_plot_template % (fish, method))
    #print("plot branch length distribution")
    #plot_branch_len_time(
    #    res.fitted_bifurc_tree,
    #    tot_time,
    #    out_plot_file="/Users/jeanfeng/Desktop/scatter_dist_to_branch_len.png")

if __name__ == "__main__":
    main()
