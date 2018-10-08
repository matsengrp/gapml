import sys
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import six
import scipy.stats
from cell_lineage_tree import CellLineageTree
from matplotlib import pyplot as plt
from scipy.stats import rankdata


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

def create_distance_matrix(fitted_bifurc_tree, obs_dict):
    organ_dict = obs_dict["organ_dict"]
    allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)

    fitted_bifurc_tree.label_dist_to_roots()
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

def plot_distance_matrix(sym_X_matrix, out_plot_file):
    for i in range(sym_X_matrix.shape[0]):
        sym_X_matrix[i,i] = 10000000000
    sym_X_matrix = np.floor(sym_X_matrix * 100)/100
    annot_mat = []
    for i in range(sym_X_matrix.shape[0]):
        new_row_annot = rankdata(sym_X_matrix[i,:])
        annot_mat.append(new_row_annot)

    # HEATMAP
    plt.clf()
    mask = np.zeros(sym_X_matrix.shape, dtype=bool)
    for i in range(sym_X_matrix.shape[0]):
        sym_X_matrix[i,i] = 0
        mask[i,i] = True
    sns.heatmap(sym_X_matrix,
            xticklabels=ORGAN_LABELS,
            yticklabels=ORGAN_LABELS,
            mask=mask,
            annot=np.array(annot_mat),
            fmt='')
    plt.savefig(out_plot_file)
    print("matrix PLOT", out_plot_file)

def load_fish(FISH, do_chronos=False):
    if FISH == "ADR1":
        obs_file = "_output/gestalt_aws/ADR1_fish_data.pkl"
    elif FISH == "ADR2":
        obs_file = "analyze_gestalt/_output/ADR2_abund1/fish_data_restrict_with_cell_types.pkl"
    if not do_chronos:
        if FISH == "ADR1":
            fitted_tree_file = "_output/gestalt_aws/ADR1_fitted.pkl"
        elif FISH == "ADR2":
            fitted_tree_file = "analyze_gestalt/_output/ADR2_abund1/sum_states_10/extra_steps_0/tune_pen_hanging.pkl"
        with open(fitted_tree_file, "rb") as f:
            if FISH == "ADR1":
                fitted_bifurc_tree = six.moves.cPickle.load(f)[0]["best_res"].fitted_bifurc_tree
            else:
                fitted_bifurc_tree = six.moves.cPickle.load(f)["final_fit"].fitted_bifurc_tree
    else:
        if FISH == "ADR1":
            fitted_tree_file = "_output/gestalt_aws/ADR1_chronos_fitted.pkl"
        elif FISH == "ADR2":
            fitted_tree_file = "analyze_gestalt/_output/ADR2_abund1/chronos_fitted.pkl"
        with open(fitted_tree_file, "rb") as f:
            fitted_bifurc_tree = six.moves.cPickle.load(f)[0]["fitted_tree"]

    with open(obs_file, "rb") as f:
        obs_dict = six.moves.cPickle.load(f)
    return fitted_bifurc_tree, obs_dict


def main(args=sys.argv[1:]):
    fishies = ["ADR1", "ADR2"]
    do_chronoses = [True, False]
    for do_chronos in do_chronoses:
        sym_X_matrices = []
        for fish in fishies:
            tree, obs_dict = load_fish(fish, do_chronos=do_chronos)
            _, sym_X_matrix = create_distance_matrix(tree, obs_dict)
            out_plot_file = "_output/sym_heat_%s%s.png" % (fish, "_chronos" if do_chronos else "")
            plot_distance_matrix(sym_X_matrix, out_plot_file)
            print(sym_X_matrix)
            sym_X_matrices.append(sym_X_matrix)

        triu_indices = np.triu_indices(NUM_ORGANS, k=1)
        print("DO CHRON", do_chronos)
        print(scipy.stats.pearsonr(sym_X_matrices[0][triu_indices], sym_X_matrices[1][triu_indices]))

if __name__ == "__main__":
    main()
