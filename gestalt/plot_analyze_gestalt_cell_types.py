import sys
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import seaborn as sns
import numpy as np
import scipy.stats
import pandas as pd
from scipy import stats

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
        default=1000)
    parser.add_argument(
        '--out-germ-layer-plot-template',
        type=str,
        default="_output/time_to_germ_layer_%s.png")
    parser.add_argument(
        '--out-cell-type-plot-template',
        type=str,
        default="_output/time_to_cell_type_%s.png")
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
            germ_layers = list(allele_to_cell_state[allele_str].keys())
            node.add_feature("germ_layers", set([
                ORGAN_GERM_LAYERS[organ_dict[c_state].replace("7B_", "")] for c_state in germ_layers]))
        else:
            node.add_feature("germ_layers", set())
            for child in node.children:
                node.germ_layers.update(child.germ_layers)

    for node in fitted_bifurc_tree.traverse('postorder'):
        if node.is_leaf():
            continue
        if min([len(leaf.germ_layers) for leaf in node]) > 1:
            continue
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
    pyplot.ylabel("Number of germ layers")
    pyplot.xlabel("Distance from root")
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
        if min([len(leaf.cell_types) for leaf in node]) > 1:
            continue
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
        tree,
        allele_to_cell_state,
        organ_dict,
        get_X_Y_func,
        num_rands):
    """
    Formally, the null hypothesis is the cell types and abundances are completely
    randomly assigned to the leaves of the two trees.
    """
    corr, stupid_pval = scipy.stats.pearsonr(estimated_X_Y[0], estimated_X_Y[1])
    all_rand_corrs = []
    for _ in range(num_rands):
        shuffled_meta = create_shuffled_cell_state_abund_labels(allele_to_cell_state)
        rand_X, rand_Y = get_X_Y_func(tree, shuffled_meta, organ_dict)
        rand_corr, _ = scipy.stats.pearsonr(rand_X, rand_Y)
        all_rand_corrs.append(rand_corr)
    print('correlation', corr, 'stupid pval', stupid_pval)
    print("random mean corr", np.mean(all_rand_corrs))
    print('p-val', np.mean(np.abs(corr) < np.abs(all_rand_corrs)))

def main(args=sys.argv[1:]):
    args = parse_args(args)
    fishies = ["ADR1", "ADR2"]
    methods = ["PMLE", "chronos", "nj"]
    for method in methods:
        print("METHOD", method)
        for fish in fishies:
            print("FISH", fish)
            tree, obs_dict = load_fish(fish, method)
            tree.label_dist_to_roots()
            allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)
            organ_dict = obs_dict["organ_dict"]

            X_dists, Y_n_germ_layers = get_distance_to_num_germ_layers(
                tree,
                allele_to_cell_state,
                organ_dict)
            do_hypothesis_test(
                (X_dists, Y_n_germ_layers),
                tree,
                allele_to_cell_state,
                organ_dict,
                get_distance_to_num_germ_layers,
                args.num_rands)
            plot_distance_to_num_germ_layers(
                X_dists,
                Y_n_germ_layers,
                args.out_germ_layer_plot_template % fish)

            X_dists, Y_n_cell_states = get_distance_to_num_cell_states(
                tree,
                allele_to_cell_state,
                organ_dict)
            do_hypothesis_test(
                (X_dists, Y_n_cell_states),
                tree,
                allele_to_cell_state,
                organ_dict,
                get_distance_to_num_cell_states,
                args.num_rands)
            plot_distance_to_num_cell_states(
                X_dists,
                Y_n_cell_states,
                args.out_cell_type_plot_template % fish)
    #print("plot branch length distribution")
    #plot_branch_len_time(
    #    res.fitted_bifurc_tree,
    #    tot_time,
    #    out_plot_file="/Users/jeanfeng/Desktop/scatter_dist_to_branch_len.png")

if __name__ == "__main__":
    main()
