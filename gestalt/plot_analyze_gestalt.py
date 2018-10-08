import six
from plot_mrca_matrices import plot_tree
from matplotlib import pyplot
from scipy import stats
import numpy as np
from sklearn.manifold import MDS, TSNE
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import itertools
import seaborn as sns

import collapsed_tree
from cell_state import CellTypeTree
from cell_lineage_tree import CellLineageTree
from common import assign_rand_tree_lengths

def rand_jitter(arr, scaling_factor=0.003):
    stdev = scaling_factor*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

ORGAN_COLORS = {
    "7B_Brain": "DarkGreen",
    "7B_Eye1": "MediumSeaGreen", # left eye
    "7B_Eye2": "LightGreen",
    "7B_Gills": "Gold",
    "7B_Intestine": "MediumBlue", # intestinal bulb
    "7B_Upper_GI": "DarkBlue", # post. intestine
    "7B_Blood": "Red",
    "7B_Heart_chunk": "Maroon",
    "7B_Heart_diss": "FireBrick", # DHC
    "7B_Heart_GFP-": "LightCoral", # NC
    "7B_Heart_GFP+": "Pink", # cardiomyocytes
}
ORGAN_TRANSLATION = {
    "Brain": "Brain",
    "Eye1": "Left eye",
    "Eye2": "Right eye",
    "Gills": "Gills",
    "Intestine": "Intestinal bulb",
    "Upper_GI": "Post intestine",
    "Blood": "Blood",
    "Heart_chunk": "Heart",
    "Heart_diss": "DHC",
    "Heart_GFP-": "NC",
    "Heart_GFP+": "Cardiomyocytes",
}
ORGAN_ORDER = {
    "Brain": 0,
    "Eye1": 1,
    "Eye2": 2,
    "Gills": 3,
    "Intestine": 4,
    "Upper_GI": 5,
    "Blood": 6,
    "Heart_GFP-": 7,
    "Heart_diss": 8,
    "Heart_GFP+": 9,
    "Heart_chunk": 10,
}
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

root_cell_type = CellTypeTree(11)
ecto_cell_type = CellTypeTree(12)
root_cell_type.add_child(ecto_cell_type)
brain_cell_type = CellTypeTree(0)
left_eye_cell_type = CellTypeTree(1)
right_eye_cell_type = CellTypeTree(2)
ecto_cell_type.add_child(brain_cell_type)
ecto_cell_type.add_child(left_eye_cell_type)
ecto_cell_type.add_child(right_eye_cell_type)
gills_cell_type = CellTypeTree(3)
root_cell_type.add_child(gills_cell_type)
endo_cell_type = CellTypeTree(13)
root_cell_type.add_child(endo_cell_type)
intestine_bulb_cell_type = CellTypeTree(4)
post_intestine_cell_type = CellTypeTree(5)
endo_cell_type.add_child(intestine_bulb_cell_type)
endo_cell_type.add_child(post_intestine_cell_type)
meso_cell_type = CellTypeTree(14)
root_cell_type.add_child(meso_cell_type)
blood_cell_type = CellTypeTree(6)
h1_cell_type = CellTypeTree(7)
h2_cell_type = CellTypeTree(8)
h3_cell_type = CellTypeTree(9)
h4_cell_type = CellTypeTree(10)
endo_cell_type.add_child(blood_cell_type)
endo_cell_type.add_child(h1_cell_type)
endo_cell_type.add_child(h2_cell_type)
endo_cell_type.add_child(h3_cell_type)
endo_cell_type.add_child(h4_cell_type)

for c in root_cell_type.get_descendants():
    c.dist = 1
gills_cell_type.dist = 2

cell_type_leaves = [
brain_cell_type,
left_eye_cell_type,
right_eye_cell_type,
gills_cell_type,
intestine_bulb_cell_type,
post_intestine_cell_type,
blood_cell_type,
h1_cell_type,
h2_cell_type,
h3_cell_type,
h4_cell_type,
]
cell_dist_mat = np.zeros((11,11))
for i in range(11):
    for j in range(11):
        if i != j:
            cell_dist_mat[i,j] = cell_type_leaves[i].get_distance(cell_type_leaves[j])

THRES = 5
COLLAPSE_DIST = 0.001

FISH = "ADR1"

if FISH == "ADR1":
    fitted_tree_file = "_output/gestalt_aws/ADR1_fitted.pkl"
    chronos_tree_file = "_output/gestalt_aws/ADR1_chronos_fitted.pkl"
    obs_file = "_output/gestalt_aws/ADR1_fish_data.pkl"
elif FISH == "ADR2":
    fitted_tree_file = "tmp_mount/analyze_gestalt/_output/ADR2_abund1/sum_states_10/extra_steps_0/tune_pen_hanging.pkl"
    chronos_tree_file = "tmp_mount/analyze_gestalt/_output/ADR2_abund1/chronos_fitted.pkl"
    obs_file = "tmp_mount/analyze_gestalt/_output/ADR2_abund1/fish_data_restrict_with_cell_types.pkl"
elif FISH.startswith("dome"):
    fitted_tree_file = "tmp_mount/analyze_gestalt/_output/%s_abund1/sum_states_10/extra_steps_0/tune_pen.pkl" % FISH
    chronos_tree_file = "tmp_mount/analyze_gestalt/_output/%s_abund1/chronos_fitted.pkl" % FISH
    obs_file = "tmp_mount/analyze_gestalt/_output/%s_abund1/fish_data_restrict.pkl" % FISH

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
                robust=True,
                )
        pyplot.xlabel("dist to root")
        pyplot.ylabel("log_2(abundance/max_abundance)")
        pyplot.xlim(-0.05,1)
        pyplot.savefig(out_plot_file)
    fitted_slope, _, fitted_corr, pval, _ = stats.linregress(X_dists, np.log2(Y_abundance))
    print("mle tree", stats.linregress(X_dists, np.log2(Y_abundance)))

def plot_distance_to_num_germ_layers(
        fitted_bifurc_tree,
        organ_dict,
        rand_tree,
        tot_time,
        out_plot_file = "/Users/jeanfeng/Desktop/scatter_dist_to_cell_state.png"):
    """
    """
    X_dists = []
    Y_n_germ_layers = []
    for node in fitted_bifurc_tree.traverse('postorder'):
        dist = node.get_distance(fitted_bifurc_tree)
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
        dist = node.get_distance(fitted_bifurc_tree)
        X_dists.append(dist)
        n_germ_layers = len(node.germ_layers)
        Y_n_germ_layers.append(n_germ_layers)

    if out_plot_file:
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
    print(stats.linregress(X_dists, Y_n_germ_layers))

def plot_distance_to_num_cell_states(
        fitted_bifurc_tree,
        organ_dict,
        rand_tree,
        tot_time,
        out_plot_file = "/Users/jeanfeng/Desktop/scatter_dist_to_cell_state.png",
        num_rands = 5):
    """
    Understand if distance to root is inversely related to number of
    different cell states in leaves
    """
    for node in fitted_bifurc_tree.traverse('postorder'):
        dist = node.get_distance(fitted_bifurc_tree)
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
        dist = node.get_distance(fitted_bifurc_tree)
        X_dists.append(dist)
        Y_n_cell_states.append(len(node.cell_types))

    if out_plot_file:
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
    print(stats.linregress(X_dists, Y_n_cell_states))

def plot_majority_cell_appearance_time(
        fitted_bifurc_tree,
        allele_to_cell_state,
        organ_dict,
        out_plot_file = "/Users/jeanfeng/Desktop/cell_appearance_time.png",
        THRES_PROPORTION = .75):
    """
    Plot the time of the progenitor cell of most of a particular cell type
    """
    total_cell_type_nums = {k: 0 for k in organ_dict.keys()}

    for leaf in fitted_bifurc_tree:
        for cell_type, abund in allele_to_cell_state[leaf.allele_events_list_str].items():
            total_cell_type_nums[cell_type] += abund
    print(total_cell_type_nums)

    organ_time = {}
    for leaf in fitted_bifurc_tree: #.traverse("preorder"):
        #node_dist = node.get_distance(fitted_bifurc_tree)
        #node_dist = leaf.up.get_distance(fitted_bifurc_tree)
        leaf_cell_type_nums = {k: 0 for k in organ_dict.keys()}
        #for leaf in node:
        for cell_type, abund in allele_to_cell_state[leaf.allele_events_list_str].items():
            leaf_cell_type_nums[cell_type] += abund

        for cell_type in organ_dict.keys():
            tot_abund = leaf_cell_type_nums[cell_type]
            if cell_type in organ_time:
                organ_time[cell_type].append((
                    cell_type,
                    tot_abund,
                    leaf))
            else:
                organ_time[cell_type] = [(
                    cell_type,
                    tot_abund,
                    leaf)]

    organ_time_list = []
    for cell_type, matches in organ_time.items():
        if total_cell_type_nums[cell_type] <= 0:
            continue
        sorted_matches = sorted(matches, key=lambda m: m[1], reverse=True)
        mrca = None
        tot_abund = 0
        for match in sorted_matches:
            if mrca is None:
                mrca = match[-1]
            else:
                mrca = mrca.get_common_ancestor(match[-1])
            tot_abund += match[1]
            if tot_abund/float(total_cell_type_nums[cell_type]) > THRES_PROPORTION:
                break
        organ_time_list.append((
            cell_type,
            mrca.get_distance(fitted_bifurc_tree),
            tot_abund/float(total_cell_type_nums[cell_type]),
            tot_abund))


    sorted_organ_times = sorted(organ_time_list, key=lambda m: m[1])
    print("all ranked")
    for organ_time in sorted_organ_times:
        if organ_time[-1] > 50:
            print(
                ORGAN_TRANSLATION[organ_dict[organ_time[0]].replace("7B_", "")],
                organ_dict[organ_time[0]],
                organ_time)

    #print("uniqify")
    #printed_organs = set()
    #for organ_time in sorted_organ_times:
    #    if organ_time[2] < 0.99999:
    #        if organ_time[0] not in printed_organs:
    #            print(
    #                ORGAN_TRANSLATION[organ_dict[organ_time[0]]],
    #                organ_dict[organ_time[0]],
    #                organ_time)
    #            printed_organs.add(organ_time[0])
def plot_mds_by_cell_type_for_tree(
    tree,
    allele_to_cell_state,
    organ_dict,
    out_plot_file,
    min_abund = 20,
    min_cell_types = 5):
    """
    Plot MDS of cell type distances
    """
    leaves_by_cell_type = {v: [] for v in organ_dict.values()}
    for leaf in tree:
        for cell_type, abund in allele_to_cell_state[leaf.allele_events_list_str].items():
            leaves_by_cell_type[organ_dict[cell_type]].append((leaf, abund))

    cell_type_list = list(organ_dict.values())
    cell_type_list = [
            cell_type for cell_type in cell_type_list
            if sum([m[1] for m in leaves_by_cell_type[cell_type]]) > min_abund]
    cell_type_list = sorted(cell_type_list, key=lambda x: ORGAN_ORDER[x.replace("7B_", "")])
    if len(cell_type_list) <= min_cell_types:
        #print("NOPE -- didnt find cell types")
        return None, None

    X_matrix = np.zeros((len(cell_type_list), len(cell_type_list)))
    for idx0, cell_type0 in enumerate(cell_type_list):
        for idx1, cell_type1 in enumerate(cell_type_list):
            if idx0 <= idx1:
                tot_abund0 = sum([m[1] for m in leaves_by_cell_type[cell_type0]])
                tot_abund1 = sum([m[1] for m in leaves_by_cell_type[cell_type1]])
                tot_dists = 0
                for leaf0, abund0 in leaves_by_cell_type[cell_type0]:
                    for leaf1, abund1 in leaves_by_cell_type[cell_type1]:
                        dist = leaf0.get_distance(leaf1)/2
                        if dist == 0:
                            dist = leaf0.dist
                        tot_dists += dist * abund0 * abund1
                #print(cell_type0, cell_type1, tot_dists/(tot_abund0 * tot_abund1), tot_abund0, tot_abund1)
                X_matrix[idx0, idx1] = tot_dists/(tot_abund0 * tot_abund1)
                X_matrix[idx1, idx0] = tot_dists/(tot_abund0 * tot_abund1)

    match_indices = [
            ORGAN_ORDER[cell_type.replace("7B_", "")]
            for cell_type in cell_type_list]
    matching_cell_dist_mat = cell_dist_mat[match_indices,:][:,match_indices]
    if (np.unique(matching_cell_dist_mat.flatten())).size > 1 and (np.unique(X_matrix.flatten())).size > 1:
        print(spearmanr(matching_cell_dist_mat.flatten(), X_matrix.flatten()))
        print(kendalltau(matching_cell_dist_mat.flatten(), X_matrix.flatten()))
        return (kendalltau(matching_cell_dist_mat.flatten(), X_matrix.flatten()))
    else:
        return None, None

    #pyplot.clf()
    #mask = np.ones(X_matrix.shape, dtype=bool)
    #mask[np.triu_indices_from(mask)] = False
    #cell_type_labels = [
    #        ORGAN_TRANSLATION[cell_type.replace("7B_", "")]
    #        for cell_type in cell_type_list]
    #sns.heatmap(X_matrix, xticklabels=cell_type_labels, yticklabels=cell_type_labels, mask=mask)
    #pyplot.savefig(out_plot_file)
    #print("matrix PLOT", out_plot_file)

    #mds = MDS(n_components=2, metric=False, max_iter=3000, eps=1e-9, dissimilarity="precomputed")
    #noise = np.random.rand(X_matrix.shape[0], X_matrix.shape[1]) * 0.02
    #X_matrix += (noise + noise.T)/2
    #pos = mds.fit(X_matrix).embedding_
    #pyplot.clf()
    #fig, ax = pyplot.subplots()
    #ax.scatter(pos[:, 0], pos[:, 1], color='turquoise', lw=0, label='MDS')
    #for i, cell_type in enumerate(cell_type_list):
    #    ax.annotate(ORGAN_TRANSLATION[cell_type.replace("7B_", "")], (pos[i, 0], pos[i, 1]))

    #pyplot.savefig(out_plot_file)
    #print("MDS PLOT", out_plot_file)

def plot_mds_by_cell_type(
    fitted_bifurc_tree,
    rand_tree,
    allele_to_cell_state,
    organ_dict,
    tot_time,
    min_abund = 0,
    out_plot_prefix = "/Users/jeanfeng/Desktop/mds"):
    """
    plot mds with cell types based on MRCA distance
    """
    ## MDS for fitted tree -- subtrees
    #for idx, node in enumerate(fitted_bifurc_tree.get_descendants()):
    #    node_dist = node.get_distance(fitted_bifurc_tree)
    #    if len(node.children) == 0:
    #        continue
    #    children_dist = min([child.get_distance(fitted_bifurc_tree) for child in node.children])
    #    if node_dist < 0.15 and children_dist > 0.15:
    #        plot_mds_by_cell_type_for_tree(
    #            node,
    #            allele_to_cell_state,
    #            organ_dict,
    #            out_plot_file = "%s_fitted_%d.png" % (out_plot_prefix, idx))

    ## MDS for rand tree -- subtrees
    #for idx, node in enumerate(rand_tree.get_children()):
    #    node_dist = node.get_distance(rand_tree)
    #    if len(node.children) == 0:
    #        continue
    #    children_dist = min([child.get_distance(rand_tree) for child in node.children])
    #    plot_mds_by_cell_type_for_tree(
    #        node,
    #        allele_to_cell_state,
    #        organ_dict,
    #        out_plot_file = "%s_rand_%d.png" % (out_plot_prefix, idx))

    # Overall MDS for fitted
    plot_mds_by_cell_type_for_tree(
        fitted_bifurc_tree,
        allele_to_cell_state,
        organ_dict,
        min_abund=min_abund,
        out_plot_file = "%s_fitted.png" % out_plot_prefix)

    # Overall MDS for rand tree
    assign_rand_tree_lengths(rand_tree, tot_time)

    plot_mds_by_cell_type_for_tree(
        rand_tree,
        allele_to_cell_state,
        organ_dict,
        out_plot_file = "%s_rand.png" % out_plot_prefix)

def plot_tsne_by_taxon_for_tree(
    tree,
    allele_to_cell_state,
    organ_dict,
    tot_time,
    out_plot_file):
    """
    Plot MDS of taxon distances
    """
    abund_by_cell_type = {v: 0 for v in organ_dict.values()}
    for leaf in tree:
        for cell_type, abund in allele_to_cell_state[leaf.allele_events_list_str].items():
            abund_by_cell_type[organ_dict[cell_type]] += abund

    tree = _expand_leaved_tree(tree, allele_to_cell_state, default_dist_scale = 0.5, min_abund_thres = 5)

    X_matrix = np.zeros((len(tree), len(tree)))
    colors = []
    sizes = []
    for idx0, leaf0 in enumerate(tree):
        colors.append(ORGAN_COLORS[organ_dict[str(leaf0.cell_state)]])
        leaf0.add_feature("leaf_id", idx0)
        sizes.append(leaf0.abundance/abund_by_cell_type[organ_dict[str(leaf0.cell_state)]])

    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            leaf_lists = []
            for c in node.children:
                leaf_list = [leaf.leaf_id for leaf in c]
                leaf_lists.append(leaf_list)
            leaf_pairs = []
            for list_idx0, leaf_list0 in enumerate(leaf_lists):
                for idx1, leaf_list1 in enumerate(leaf_lists[list_idx0 + 1:]):
                    list_idx1 = list_idx0 + idx1 + 1
                    leaf_pairs += list(itertools.product(leaf_list0, leaf_list1))

            #node.add_feature("mrca_pairs", leaf_pairs)

            dist = tot_time - node.get_distance(tree)
            for (leaf_idx0, leaf_idx1) in leaf_pairs:
                X_matrix[leaf_idx0, leaf_idx1] = dist
                X_matrix[leaf_idx1, leaf_idx0] = dist

    print("embedding....")
    mds = TSNE(n_components=2, metric="precomputed")
    noise = np.random.rand(X_matrix.shape[0], X_matrix.shape[1]) * 0.02
    X_matrix += (noise + noise.T)/2
    pos = mds.fit(X_matrix).embedding_
    pyplot.clf()
    fig, ax = pyplot.subplots()
    ax.scatter(pos[:, 0], pos[:, 1], color=colors, lw=0, alpha = 0.5, s=np.array(sizes) * 1000)

    pyplot.savefig(out_plot_file)
    print("TSNE PLOT", out_plot_file)

def plot_tsne_by_taxon(
    fitted_bifurc_tree,
    rand_tree,
    allele_to_cell_state,
    organ_dict,
    tot_time,
    out_plot_prefix = "/Users/jeanfeng/Desktop/tsne"):
    """
    plot TSNE of all cells based on MRCA distance
    """
    assign_rand_tree_lengths(rand_tree, tot_time)
    plot_tsne_by_taxon_for_tree(
        fitted_bifurc_tree,
        allele_to_cell_state,
        organ_dict,
        tot_time,
        out_plot_file = "%s_taxon_fitted.png" % out_plot_prefix)
    plot_tsne_by_taxon_for_tree(
        rand_tree,
        allele_to_cell_state,
        organ_dict,
        tot_time,
        out_plot_file = "%s_taxon_rand.png" % out_plot_prefix)

def plot_branch_len_time(
    fitted_bifurc_tree,
    tot_time,
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

def _expand_leaved_tree(fitted_bifurc_tree, allele_to_cell_state, default_dist_scale = 0, min_abund_thres = 0):
    """
    @param default_dist_scale: how much to assign the leaf branch to the different cell types vs. preserve as internal branch length
    @param min_abund_thres: minimum abundance for us to include that cell type leaf in the tree (we will always include
                        the cell type with the highest abundance, regardless of absolute abundance)
    """
    leaved_tree = fitted_bifurc_tree.copy()
    for l in leaved_tree:
        allele_str = l.allele_events_list_str
        if l.cell_state is None:
            old_dist = l.dist
            l.dist = old_dist * (1 - default_dist_scale)
            sorted_cell_states = sorted(
                    [(c_state_str, abund) for c_state_str, abund in allele_to_cell_state[allele_str].items()],
                    key = lambda c: c[1],
                    reverse=True)
            for c_state_str, abund in sorted_cell_states[:1]:
                new_child = CellLineageTree(
                    l.allele_list,
                    l.allele_events_list,
                    cell_state_dict[c_state_str],
                    dist = old_dist * default_dist_scale,
                    abundance = abund,
                    resolved_multifurcation = True)
                l.add_child(new_child)
            for c_state_str, abund in sorted_cell_states[1:]:
                if abund > min_abund_thres:
                    new_child = CellLineageTree(
                        l.allele_list,
                        l.allele_events_list,
                        cell_state_dict[c_state_str],
                        dist = old_dist * default_dist_scale,
                        abundance = abund,
                        resolved_multifurcation = True)
                    l.add_child(new_child)
    print("num leaves", len(leaved_tree))
    return leaved_tree

"""
plotting my fitted tree now...
"""
def plot_gestalt_tree(
        fitted_bifurc_tree,
        organ_dict,
        allele_to_cell_state,
        cell_state_dict,
        out_plot_file):
    from ete3 import NodeStyle, SeqMotifFace
    fitted_bifurc_tree = _expand_leaved_tree(fitted_bifurc_tree, allele_to_cell_state)

    for l in fitted_bifurc_tree:
        nstyle = NodeStyle()
        nstyle["fgcolor"] = ORGAN_COLORS[organ_dict[str(l.cell_state)]]
        nstyle["size"] = 10
        l.set_style(nstyle)

    for leaf in fitted_bifurc_tree:
        # get the motif list for indels in the format that SeqMotifFace expects
        motifs = []
        for event in leaf.allele_events_list[0].events:
            motifs.append([
                event.start_pos,
                event.start_pos + len(event.insert_str),
                '[]',
                len(event.insert_str),
                10,
                'black',
                'blue',
                None
            ])
        for event in leaf.allele_events_list[0].events:
            motifs.append([
                event.start_pos,
                event.del_end,
                '[]',
                event.del_len,
                10,
                'black',
                'red',
                None
            ])
        seq = ''.join(bcode_meta.unedited_barcode)
        seqFace = SeqMotifFace(
            seq=seq.upper(),
            motifs=motifs,
            seqtype='nt',
            seq_format='[]',
            height=10,
            gapcolor='red',
            gap_format='[]',
            fgcolor='black',
            bgcolor='lightgrey')
        leaf.add_face(seqFace, 0, position="aligned")

    # Collapse distances for plot readability
    for node in fitted_bifurc_tree.get_descendants():
        if node.dist < COLLAPSE_DIST:
            node.dist = 0
    col_tree = collapsed_tree.collapse_zero_lens(fitted_bifurc_tree)

    legend_colors = {}
    for organ_key, color in ORGAN_COLORS.items():
        text = ORGAN_TRANSLATION[organ_key.replace("7B_", "")]
        label_dict = {
            "text": text,
            "color": "gray",
            "fontsize": 8}
        legend_colors[color] = label_dict

    print("at plotting phase....")
    plot_tree(
            col_tree,
            out_plot_file,
            width=400,
            show_leaf_name=False,
            legend_colors=legend_colors)

"""
Do the main things
"""
with open(obs_file, "rb") as f:
    obs_dict = six.moves.cPickle.load(f)

tot_time = obs_dict["time"]
bcode_meta = obs_dict["bcode_meta"]
organ_dict = obs_dict["organ_dict"]
allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)

with open(fitted_tree_file, "rb") as f:
    if FISH == "ADR1":
        res = six.moves.cPickle.load(f)[0]["best_res"]
    elif FISH == "ADR2" or FISH.startswith("dome"):
        res = six.moves.cPickle.load(f)["final_fit"]

with open(chronos_tree_file, "rb") as f:
    chronos_tree = six.moves.cPickle.load(f)[0]["fitted_tree"]

print("plot mds")
#plot_mds_by_cell_type(
#    res.fitted_bifurc_tree,
#    rand_tree,
#    allele_to_cell_state,
#    organ_dict,
#    tot_time,
#    min_abund=5,
#    out_plot_prefix = "/Users/jeanfeng/Desktop/dist_matrix_ADR2")
#print(len(res.fitted_bifurc_tree))
#tot_leaves = 0
#spine_children_ids = res.fitted_bifurc_tree.spine_children
#print(spine_children_ids)
#idx = 0
#all_corrs = []
#for i, spine_id in enumerate(res.fitted_bifurc_tree.spine_children[1:]):
#    spine_node = res.fitted_bifurc_tree.search_nodes(node_id=spine_id)[0]
#    nonspine_children = [c for c in spine_node.get_children() if c.node_id not in spine_children_ids]
#    for nonspine_child in nonspine_children:
#        tot_leaves += len(nonspine_child)
#        #print("NON SPINE CHILD", nonspine_child.node_id)
#        #print([c.node_id for c in nonspine_child.get_children()])
#        corr = plot_mds_by_cell_type_for_tree(
#            nonspine_child,
#            allele_to_cell_state,
#            organ_dict,
#            min_abund=10,
#            min_cell_types=5,
#            out_plot_file= "/Users/jeanfeng/Desktop/dist_matrix_ADR2_%d.png" % idx)
#        if corr[0] is not None:
#            all_corrs.append(corr[0])
#        idx += 1
#print(tot_leaves)
#print("mle", np.mean(all_corrs))
#
#all_corrs = []
#for i, chronos_child in enumerate(chronos_tree.get_children()):
#    corr = plot_mds_by_cell_type_for_tree(
#        chronos_child,
#        allele_to_cell_state,
#        organ_dict,
#        min_abund=5,
#        min_cell_types=3,
#        out_plot_file = "/Users/jeanfeng/Desktop/dist_matrix_chronos_ADR2_%d.png" % i)
#    if corr[0] is not None:
#        all_corrs.append(corr[0])
#print("chrono", np.mean(all_corrs))
#print("distance to abundance")
#plot_distance_to_abundance(
#    res.fitted_bifurc_tree,
#    tot_time,
#    out_plot_file = "/Users/jeanfeng/Documents/Research/gestalt/gestaltamania-tex/manuscript/images/scatter_dist_to_abundance_%s.png" % FISH)
#plot_distance_to_abundance(
#    chronos_tree,
#    tot_time,
#    out_plot_file = "/Users/jeanfeng/Documents/Research/gestalt/gestaltamania-tex/manuscript/images/scatter_dist_to_abundance_chronos_%s.png" % FISH)
#print("distance to number of descendant cell states")
#plot_distance_to_num_germ_layers(
#    res.fitted_bifurc_tree,
#    organ_dict,
#    rand_tree,
#    tot_time,
#    out_plot_file = "/Users/jeanfeng/Desktop/scatter_dist_to_germ_%s.png" % FISH)
#plot_distance_to_num_germ_layers(
#    chronos_tree,
#    organ_dict,
#    rand_tree,
#    tot_time,
#    out_plot_file = "/Users/jeanfeng/Desktop/scatter_dist_to_germ_chronos_%s.png" % FISH)
#plot_distance_to_num_cell_states(
#    res.fitted_bifurc_tree,
#    organ_dict,
#    rand_tree,
#    tot_time,
#    out_plot_file = "/Users/jeanfeng/Desktop/scatter_dist_to_cell_state_%s.png" % FISH)
#plot_distance_to_num_cell_states(
#    chronos_tree,
#    organ_dict,
#    rand_tree,
#    tot_time,
#    out_plot_file = "/Users/jeanfeng/Desktop/scatter_dist_to_cell_state_chronos_%s.png" % FISH)
#print("plot branch length distribution")
plot_branch_len_time(
    res.fitted_bifurc_tree,
    tot_time,
    out_plot_file="/Users/jeanfeng/Desktop/scatter_dist_to_branch_len.png")
#plot_branch_len_time(
#    chronos_tree,
#    rand_tree,
#    tot_time,
#    out_plot_file="/Users/jeanfeng/Desktop/scatter_dist_to_branch_len_chronos.png",
#    num_rands = 2000)
#plot_gestalt_tree(
#    chronos_tree,
#    organ_dict,
#    allele_to_cell_state,
#    cell_state_dict,
#    "/Users/jeanfeng/Desktop/ADR1_chronos_fitted.png")
#plot_gestalt_tree(
#    res.fitted_bifurc_tree,
#    organ_dict,
#    allele_to_cell_state,
#    cell_state_dict,
#    "/Users/jeanfeng/Desktop/ADR1_fitted.png")
