import sys
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

from common import assign_rand_tree_lengths

"""
Create distance matrices between cell types in adult fish 1 and 2
Also calculates the correlation between the distance matrices
"""

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

ORGAN_ORDER = {
    "Brain": 0,
    "Eye1": 1,
    "Eye2": 2,
    "Gills": 3,
    "Intestine": 4,
    "Upper_GI": 5,
    "Blood": 6,
    "Heart_GFP+": 7,
    "Heart_chunk": 8,
    "Heart_GFP-": 9,
    "Heart_diss": 10,
}
ORGAN_LABELS = [
    "Brain",
    "Eye1",
    "Eye2",
    "Gills",
    "Intestine",
    "Upper_GI",
    "Blood",
    "Heart_GFP+",
    "Heart_chunk",
    "Heart_GFP-",
    "Heart_diss",
]
NUM_ORGANS = len(ORGAN_LABELS)

def create_distance_matrix(fitted_bifurc_tree, organ_dict, allele_to_cell_state):
    for node in fitted_bifurc_tree.traverse('postorder'):
        if node.is_leaf():
            allele_str = node.allele_events_list_str
            cell_types = allele_to_cell_state[allele_str].keys()
            node.add_feature(
                "cell_types",
                set([ORGAN_ORDER[organ_dict[x].replace("7B_", "")] for x in cell_types]))
            #assert len(cell_types) == len(node.cell_types)
        else:
            node.add_feature("cell_types", set())
            for child in node.children:
                node.cell_types.update(child.cell_types)

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

def plot_distance_matrix(sym_X_matrix, out_plot_file, vrange = [0.2, 0.8]):
    for i in range(sym_X_matrix.shape[0]):
        sym_X_matrix[i,i] = 10000000000
    sym_X_matrix = np.floor(sym_X_matrix * 100)/100
    #annot_mat = []
    #for i in range(sym_X_matrix.shape[0]):
    #    new_row_annot = rankdata(sym_X_matrix[i,:])
    #    annot_mat.append(new_row_annot)

    # HEATMAP
    plt.clf()
    mask = np.zeros(sym_X_matrix.shape, dtype=bool)
    mask[np.triu_indices(sym_X_matrix.shape[0], k=0)] = True
    #print(np.min(sym_X_matrix[np.triu_indices(sym_X_matrix.shape[0], k=1)]))
    #assert np.max(sym_X_matrix) < VMAX
    sns.heatmap(sym_X_matrix,
            xticklabels=ORGAN_LABELS,
            yticklabels=ORGAN_LABELS,
            mask=mask,
            #annot=np.array(annot_mat),
            #fmt='',
            vmin=vrange[0],
            vmax=vrange[1])
    plt.savefig(out_plot_file, transparent=True, bbox_inches='tight')
    print("matrix PLOT", out_plot_file)

def load_fish(fish, method, folder=None, get_first=False):
    obs_file = "analyze_gestalt/_output/%s/sampling_seed0/fish_data_restrict.pkl" % fish

    if method == "PMLE":
        fitted_tree_file = "analyze_gestalt/_output/%s/sampling_seed0/sum_states_25/extra_steps_1/tune_pen_hanging.pkl" % fish
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
        fitted_tree_file = "analyze_gestalt/_output/%s/sampling_seed0/chronos_fitted.pkl" % fish
        #if fish == "ADR1":
        #    fitted_tree_file = "analyze_gestalt/_output/ADR1_abund5/chronos_fitted.pkl"
        #elif fish == "ADR2":
        #    fitted_tree_file = "analyze_gestalt/_output/ADR2_abund1/chronos_fitted.pkl"
        if folder is not None:
            fitted_tree_file = os.path.join(folder, fitted_tree_file)
        with open(fitted_tree_file, "rb") as f:
            fitted_bifurc_tree = six.moves.cPickle.load(f)[0]["fitted_tree"]
    elif method == "nj":
        fitted_tree_file = "analyze_gestalt/_output/%s/sampling_seed0/nj_fitted.pkl" % fish
        #if fish == "ADR1":
        #    fitted_tree_file = "analyze_gestalt/_output/ADR1_abund5/nj_fitted.pkl"
        #elif fish == "ADR2":
        #    fitted_tree_file = "analyze_gestalt/_output/ADR2_abund1/nj_fitted.pkl"
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
    num_rand_permute = 2000
    null_method = None
    random.seed(0)
    np.random.seed(0)
    fishies = ["ADR1", "ADR2"]
    methods = ["PMLE", "chronos", "nj"]
    method_plotting_values = {
            "PMLE": [0.2, 0.8],
            "chronos": [0, 0.6],
            "nj": [0, 0.35],
    }
    for method in methods:
        sym_X_matrices = []
        random_permute_X_matrices = [[],[]]
        for fish_idx, fish in enumerate(fishies):
            print("FISH", fish)
            null_method = method if null_method is None else null_method
            null_tree, _ = load_fish(fish, null_method)

            tree, obs_dict = load_fish(fish, method)
            tree.label_dist_to_roots()
            organ_dict = obs_dict["organ_dict"]
            allele_to_cell_state, _ = get_allele_to_cell_states(obs_dict)
            _, sym_X_matrix = create_distance_matrix(tree, organ_dict, allele_to_cell_state)
            out_plot_file = "_output/sym_heat_%s_%s.png" % (fish, method)
            plot_distance_matrix(sym_X_matrix, out_plot_file, method_plotting_values[method])
            sym_X_matrices.append(sym_X_matrix)

            # Now compare against the "null distribution" where the cell type and abundance
            # labels are both shuffled. The null distribution is shared across all different tree fitting methods
            # In particular, shuffle the cell type labels (keeping them grouped) across the leaves.
            # Then shuffle the abundances within that cell type label group.
            for _ in range(num_rand_permute):
                allele_to_cell_state_random = create_shuffled_cell_state_abund_labels(allele_to_cell_state)
                assign_rand_tree_lengths(null_tree, 1)
                null_tree.label_dist_to_roots()

                _, rand_permut_X_matrix = create_distance_matrix(
                        null_tree,
                        organ_dict,
                        allele_to_cell_state_random)
                random_permute_X_matrices[fish_idx].append(rand_permut_X_matrix)

        print(method)
        do_hypothesis_test(sym_X_matrices, random_permute_X_matrices)

if __name__ == "__main__":
    main()
