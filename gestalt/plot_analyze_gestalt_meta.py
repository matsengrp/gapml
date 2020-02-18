import sys
import argparse
import random
import matplotlib
matplotlib.use('Agg')
import os.path
import seaborn as sns
import numpy as np
import six
import scipy.stats
from cell_lineage_tree import CellLineageTree
from matplotlib import pyplot as plt
from scipy.stats import rankdata

from common import assign_rand_tree_lengths, parse_comma_str

"""
Create distance matrices between cell types in adult fish 1 and 2
Also calculates the correlation between the distance matrices
"""

FILTER_BLOOD = True
ORGAN_ORDER = {
    "Brain": 0,
    "Eye1": 1,
    "Eye2": 2,
    "Gills": 3,
    "Intestine": 4,
    "Upper_GI": 5,
    "Heart_GFP+": 6,
    "Heart_chunk": 7,
    "Heart_GFP-": 8,
    "Heart_diss": 9,
    "Blood": 10,
}
ORGAN_LABELS = [
    "Brain",
    "Left Eye",
    "Right Eye",
    "Gills",
    "Intestinal bulb",
    "Post intestine",
    "Cardiomyocytes",
    "Heart",
    "NC",
    "DHC",
    "Blood",
]
NUM_ORGANS = len(ORGAN_LABELS) - int(FILTER_BLOOD)
METHOD_PLOTTING_VALS = {
        "PMLE": [0.2, 0.8],
        "chronos": [0, 0.6],
        "nj": [0, 0.35],
}

def parse_args(args):
    parser = argparse.ArgumentParser(
            description="""
            compare tissue distances between adult fish.
            """)
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
        '--num-rand-permute',
        type=int,
        default=2000)
    parser.add_argument(
        '--null-method',
        type=str,
        default=None,
        help="If none, use the method itself as reference. Otherwise always use this method as reference.")
    parser.add_argument(
        '--fishies',
        type=str,
        default="ADR1,ADR2",
        help="fish to compare, comma separated")
    parser.add_argument(
        '--methods',
        type=str,
        default="PMLE,chronos,nj",
        help="methods to compare, comma separated")
    parser.add_argument(
        '--out-fish-plot-template',
        type=str,
        help="The path to store plot file for each fish and method.")
    parser.add_argument(
        '--out-plot-template',
        type=str,
        help="The path to store plot files for each method (each plot has all fish).")
    args = parser.parse_args(args)

    args.fishies = parse_comma_str(args.fishies, str)
    args.methods = parse_comma_str(args.methods, str)
    for m in args.methods:
        assert m in METHOD_PLOTTING_VALS.keys()
    return args

def get_allele_to_cell_states(obs_dict):
    # Create allele string to cell state
    allele_to_cell_state = {}
    cell_state_dict = {}
    for obs in obs_dict["obs_leaves_by_allele_cell_state"]:
        allele_str_key = CellLineageTree._allele_list_to_str(obs.allele_events_list)
        if allele_str_key in allele_to_cell_state:
            if str(obs.cell_state) not in allele_to_cell_state:
                allele_to_cell_state[allele_str_key][str(obs.cell_state)] = obs.abundance
        else:
            allele_to_cell_state[allele_str_key] = {str(obs.cell_state): obs.abundance}

        if str(obs.cell_state) not in cell_state_dict:
            cell_state_dict[str(obs.cell_state)] = obs.cell_state

    return allele_to_cell_state, cell_state_dict

def create_distance_matrix(fitted_bifurc_tree, organ_dict, allele_to_cell_state, filter_blood=False):
    fitted_bifurc_tree = fitted_bifurc_tree.copy()
    if filter_blood:
        for leaf in fitted_bifurc_tree:
            allele_str = leaf.allele_events_list_str
            cell_types = allele_to_cell_state[allele_str].keys()
            cell_types = [organ_dict[x].replace("7B_", "") for x in cell_types]
            if "Blood" in cell_types:
                leaf.delete(preserve_branch_length=True)

    for node in fitted_bifurc_tree.traverse('postorder'):
        if node.is_leaf():
            allele_str = node.allele_events_list_str
            cell_types = allele_to_cell_state[allele_str].keys()
            node.add_feature(
                "cell_types",
                set([ORGAN_ORDER[organ_dict[x].replace("7B_", "")] for x in cell_types]))
            #print(set([organ_dict[x].replace("7B_", "") for x in cell_types]))
            #assert len(cell_types) == len(node.cell_types)
        else:
            node.add_feature("cell_types", set())
            for child in node.children:
                node.cell_types.update(child.cell_types)
        assert 10 not in node.cell_types

    X_matrix = np.zeros((NUM_ORGANS,NUM_ORGANS))
    tot_path_abundances = np.zeros((NUM_ORGANS, NUM_ORGANS))
    for leaf in fitted_bifurc_tree:
        allele_str = leaf.allele_events_list_str
        cell_type_abund_dict = allele_to_cell_state[allele_str]
        observed_cell_types = set()
        node_abund_dict = dict()
        for cell_type, abund in cell_type_abund_dict.items():
            node_cell_type = ORGAN_ORDER[organ_dict[cell_type].replace("7B_", "")]
            node_abund_dict[node_cell_type] = abund

        up_node = leaf
        while not up_node.is_root() and len(observed_cell_types) < NUM_ORGANS:
            diff_cell_types = up_node.cell_types - observed_cell_types
            for up_cell_type in diff_cell_types:
                for leaf_cell_type, abund in node_abund_dict.items():
                    if leaf_cell_type == up_cell_type:
                        continue
                    if up_node.is_leaf():
                        dist_to_root = 1 - up_node.dist/2
                    else:
                        dist_to_root = up_node.dist_to_root
                    X_matrix[leaf_cell_type, up_cell_type] += (1 - dist_to_root) * abund
                    tot_path_abundances[leaf_cell_type, up_cell_type] += abund
            observed_cell_types.update(diff_cell_types)
            up_node = up_node.up

    X_matrix = X_matrix/tot_path_abundances
    sym_X_matrix = (X_matrix + X_matrix.T)/2
    return X_matrix, sym_X_matrix

def plot_two_distance_matrices(sym_X_matrices, out_plot_file=None, vrange = [0.2, 0.8]):
    plt.clf()
    fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14,7))
    cbar_ax = fig.add_axes([.91, .3, .03, .6])
    for i, ax in enumerate(axn.flat):
        plot_distance_matrix(sym_X_matrices[i],
                ax=ax,
                cbar=i == 0,
                cbar_ax=None if i else cbar_ax,
                vrange=vrange)

    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(out_plot_file, transparent=True, bbox_inches='tight')
    print("matrix PLOT", out_plot_file)

def plot_distance_matrix(sym_X_matrix, out_plot_file=None, ax=None, cbar=True, cbar_ax=None, vrange = [0.2, 0.8]):
    for i in range(sym_X_matrix.shape[0]):
        sym_X_matrix[i,i] = 10000000000
    sym_X_matrix = np.floor(sym_X_matrix * 100)/100
    #annot_mat = []
    #for i in range(sym_X_matrix.shape[0]):
    #    new_row_annot = rankdata(sym_X_matrix[i,:])
    #    annot_mat.append(new_row_annot)

    # HEATMAP
    mask = np.zeros(sym_X_matrix.shape, dtype=bool)
    mask[np.triu_indices(sym_X_matrix.shape[0], k=0)] = True
    #print(np.min(sym_X_matrix[np.triu_indices(sym_X_matrix.shape[0], k=1)]))
    #assert np.max(sym_X_matrix) < VMAX
    sns.heatmap(sym_X_matrix,
            xticklabels=ORGAN_LABELS[:NUM_ORGANS],
            yticklabels=ORGAN_LABELS[:NUM_ORGANS],
            mask=mask,
            #annot=np.array(annot_mat),
            #fmt='',
            ax=ax,
            cbar=cbar,
            cbar_ax=cbar_ax,
            vmin=vrange[0],
            vmax=vrange[1])
    if out_plot_file:
        plt.savefig(out_plot_file, transparent=True, bbox_inches='tight')
        print("matrix PLOT", out_plot_file)

def load_fish(fish, args, method, folder=None, get_first=False):
    obs_file = args.obs_file % fish

    if method == "PMLE":
        fitted_tree_file = args.mle_template % fish
        if folder is not None:
            fitted_tree_file = os.path.join(folder, fitted_tree_file)
        with open(fitted_tree_file, "rb") as f:
            if not get_first:
                fitted_bifurc_tree = six.moves.cPickle.load(f)["final_fit"].fitted_bifurc_tree
            else:
                tune_hist = six.moves.cPickle.load(f)["tuning_history"]
                for hist_iter in tune_hist:
                    print("...")
                    if hist_iter["chad_tune_result"].num_chad_leaves == 1:
                        fitted_bifurc_tree = hist_iter["chad_tune_result"].get_best_result()[-1].fitted_bifurc_tree
                        break

    elif method == "chronos":
        fitted_tree_file = args.chronos_template % fish
        if folder is not None:
            fitted_tree_file = os.path.join(folder, fitted_tree_file)
        with open(fitted_tree_file, "rb") as f:
            fitted_bifurc_tree = six.moves.cPickle.load(f)[0]["fitted_tree"]
    elif method == "nj":
        fitted_tree_file = args.nj_template % fish
        if folder is not None:
            fitted_tree_file = os.path.join(folder, fitted_tree_file)
        with open(fitted_tree_file, "rb") as f:
            fitted_bifurc_tree = six.moves.cPickle.load(f)[1]["fitted_tree"]
    else:
        raise ValueError("method not known")

    if folder is not None:
        obs_file = os.path.join(folder, obs_file)
    with open(obs_file, "rb") as f:
        obs_dict = six.moves.cPickle.load(f)
    return fitted_bifurc_tree, obs_dict

def do_hypothesis_test(estimated_matrices, random_matrices):
    """
    Formally, the null hypothesis is the cell types and abundances are completely
    randomly assigned to the leaves of the two trees.
    """
    triu_indices = np.triu_indices(estimated_matrices[0].shape[0], k=1)
    # We cannot use the pvalue from the usual pearson correlation calculation since
    # it assumes the observations are independent.
    corr, stupid_pval = scipy.stats.pearsonr(
            estimated_matrices[0][triu_indices],
            estimated_matrices[1][triu_indices])
    all_rand_corrs = []
    for rand_mat1, rand_mat2 in zip(random_matrices[0], random_matrices[1]):
        rand_corr, _ = scipy.stats.pearsonr(rand_mat1[triu_indices], rand_mat2[triu_indices])
        all_rand_corrs.append(rand_corr)
    print('correlation', corr, 'stupid pval', stupid_pval)
    print("random mean corr", np.mean(all_rand_corrs))
    print('p-val', np.mean(np.abs(corr) < np.abs(all_rand_corrs)))

def create_shuffled_cell_state_abund_labels(allele_to_cell_state):
    num_leaves = len(allele_to_cell_state)
    key_list = list(allele_to_cell_state.keys())
    allele_to_cell_state_random = {}
    shuffled_leaves = np.random.choice(num_leaves, num_leaves, replace=False)
    for rand_idx, allele_key in zip(shuffled_leaves, key_list):
        orig_leaf_dict = allele_to_cell_state[key_list[rand_idx]]
        shuffled_abunds = np.random.choice(
                list(orig_leaf_dict.values()),
                len(orig_leaf_dict),
                replace=False)
        #allele_to_cell_state_random[allele_key] = {
        #    key: val for key, val in zip(orig_leaf_dict.keys(), shuffled_abunds)}
        random_cell_types = np.random.choice(len(ORGAN_LABELS), size=len(orig_leaf_dict), replace=False)
        allele_to_cell_state_random[allele_key] = {
            str(key): val for key, val in zip(random_cell_types, shuffled_abunds)}
    return allele_to_cell_state_random

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    random.seed(0)
    np.random.seed(0)

    sns.set_context("paper", font_scale=1.7)
    for method in args.methods:
        sym_X_matrices = []
        random_permute_X_matrices = [[],[]]
        for fish_idx, fish in enumerate(args.fishies):
            print("FISH", fish)
            null_method = method if args.null_method is None else args.null_method
            null_tree, _ = load_fish(fish, args, null_method)

            tree, obs_dict = load_fish(fish, args, method)
            tree.label_dist_to_roots()
            organ_dict = obs_dict["organ_dict"]
            allele_to_cell_state, _ = get_allele_to_cell_states(obs_dict)
            _, sym_X_matrix = create_distance_matrix(tree, organ_dict, allele_to_cell_state, filter_blood=FILTER_BLOOD)
            out_plot_file = args.out_fish_plot_template % (fish, method)
            plt.clf()
            plt.figure(figsize=(6,5))
            plot_distance_matrix(
                    sym_X_matrix,
                    out_plot_file,
                    ax=None,
                    vrange=METHOD_PLOTTING_VALS[method])
            sym_X_matrices.append(sym_X_matrix)

            # Now compare against the "null distribution" where the cell type and abundance
            # labels are both shuffled. The null distribution is shared across all different tree fitting methods
            # In particular, shuffle the cell type labels (keeping them grouped) across the leaves.
            # Then shuffle the abundances within that cell type label group.
            for _ in range(args.num_rand_permute):
                allele_to_cell_state_random = create_shuffled_cell_state_abund_labels(allele_to_cell_state)
                assign_rand_tree_lengths(null_tree, 1)
                null_tree.label_dist_to_roots()

                _, rand_permut_X_matrix = create_distance_matrix(
                        null_tree,
                        organ_dict,
                        allele_to_cell_state_random,
                        filter_blood=FILTER_BLOOD)
                random_permute_X_matrices[fish_idx].append(rand_permut_X_matrix)
        out_plot_file = args.out_plot_template % method
        plot_two_distance_matrices(
                sym_X_matrices,
                out_plot_file,
                vrange=METHOD_PLOTTING_VALS[method])

        print(method)
        do_hypothesis_test(sym_X_matrices, random_permute_X_matrices)

if __name__ == "__main__":
    main()
