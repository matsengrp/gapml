import six
from ete3 import NodeStyle, SeqMotifFace
from plot_mrca_matrices import plot_tree
from matplotlib import pyplot
from scipy import stats
import numpy as np
from sklearn.manifold import MDS

import collapsed_tree
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
    "7B_Brain": "Brain",
    "7B_Eye1": "Left eye",
    "7B_Eye2": "Right eye",
    "7B_Gills": "Gills",
    "7B_Intestine": "Intestinal bulb",
    "7B_Upper_GI": "Post intestine",
    "7B_Blood": "Blood",
    "7B_Heart_chunk": "Heart",
    "7B_Heart_diss": "DHC",
    "7B_Heart_GFP-": "NC",
    "7B_Heart_GFP+": "Cardiomyocytes",
}

THRES = 5
COLLAPSE_DIST = 0.001

fitted_tree_file = "tmp_mount/analyze_gestalt/_output/min_abund_5/sum_states_10/extra_steps_0/tune_topology_fitted.pkl"
rand_tree_file = "tmp_mount/analyze_gestalt/_output/min_abund_5/parsimony_tree1.pkl"
obs_file = "tmp_mount/analyze_gestalt/_output/min_abund_0/fish_data.pkl"

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
        rand_tree,
        tot_time,
        out_plot_file="/Users/jeanfeng/Desktop/scatter_dist_to_abundance.png",
        num_rands = 5):
    """
    Understand if distance to root is inversely related to total abundance
    """
    X_dists = []
    Y_abundance = []
    scatter_X_dists = []
    scatter_Y_abundance = []
    for node in fitted_bifurc_tree.traverse():
        dist = node.get_distance(res.fitted_bifurc_tree)
        X_dists.append(dist)
        tot_abundance = sum([leaf.abundance for leaf in node])
        Y_abundance.append(tot_abundance)
        jitter = (np.random.rand() - 0.5) * 0.1
        scatter_X_dists += [dist + jitter] * tot_abundance
        jitter = (np.random.rand() - 0.5) * 0.5
        scatter_Y_abundance += [tot_abundance + jitter] * tot_abundance

    if out_plot_file:
        pyplot.clf()
        pyplot.scatter(
                rand_jitter(scatter_X_dists, scaling_factor=0.002),
                rand_jitter(np.log10(scatter_Y_abundance), scaling_factor=0.002))
        pyplot.savefig(out_plot_file)
    print("mle tree", stats.linregress(X_dists, np.log10(Y_abundance)))

    rand_slopes = []
    rand_corr = []
    rand_pvals = []
    for _ in range(num_rands):
        assign_rand_tree_lengths(rand_tree, tot_time)

        X_dists = []
        Y_abundance = []
        for node in rand_tree.traverse():
            dist = node.get_distance(rand_tree)
            X_dists.append(dist)
            tot_abundance = sum([leaf.abundance for leaf in node])
            Y_abundance.append(tot_abundance)
        slope, _, corr, pval, _ = stats.linregress(X_dists, np.log10(Y_abundance))
        rand_slopes.append(slope)
        rand_corr.append(corr)
        rand_pvals.append(pval)
    print("rand tree", np.mean(rand_slopes), np.power(np.mean(rand_corr),2), np.mean(rand_pvals))

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
    X_dists = []
    Y_n_cell_states = []
    scatter_X_dists = []
    scatter_Y_n_cell_states = []
    colors = []
    for node in fitted_bifurc_tree.traverse():
        dist = node.get_distance(fitted_bifurc_tree)
        if dist < obs_dict["time"] - 0.01:
            X_dists.append(dist)
            c_state_set = set()
            for leaf in node:
                allele_str = leaf.allele_events_list_str
                cell_state_strs = list(allele_to_cell_state[allele_str].keys())
                for c_state_str in cell_state_strs:
                    c_state_set.update(c_state_str)
                    colors.append(
                            ORGAN_COLORS[organ_dict[c_state_str]])

            n_cell_states = len(c_state_set)
            Y_n_cell_states.append(n_cell_states)
            jitter = (np.random.rand() - 0.5) * 0.1
            scatter_X_dists += [dist + jitter] * n_cell_states
            jitter = (np.random.rand() - 0.5) * 0.5
            scatter_Y_n_cell_states += [n_cell_states + jitter] * n_cell_states

    if out_plot_file:
        pyplot.clf()
        pyplot.scatter(
                rand_jitter(scatter_X_dists, scaling_factor=0.002),
                rand_jitter(scatter_Y_n_cell_states, scaling_factor=0.002),
                c=colors,
                alpha=0.25,
                marker="o",
                s=10)
        pyplot.savefig(out_plot_file)
    print("mle tree", stats.linregress(X_dists, Y_n_cell_states))

    rand_slopes = []
    rand_corr = []
    rand_pvals = []
    for _ in range(num_rands):
        assign_rand_tree_lengths(rand_tree, tot_time)

        X_dists = []
        Y_n_cell_states = []
        for node in rand_tree.traverse():
            dist = node.get_distance(rand_tree)
            if dist < obs_dict["time"] - 0.01:
                X_dists.append(dist)
                c_state_set = set()
                for leaf in node:
                    allele_str = leaf.allele_events_list_str
                    cell_state_strs = list(allele_to_cell_state[allele_str].keys())
                    for c_state_str in cell_state_strs:
                        c_state_set.update(c_state_str)
                Y_n_cell_states.append(len(c_state_set))
        slope, _, corr, pval, _ = stats.linregress(X_dists, Y_n_cell_states)
        rand_slopes.append(slope)
        rand_corr.append(corr)
        rand_pvals.append(pval)
    print("rand tree", np.mean(rand_slopes), np.power(np.mean(rand_corr),2), np.mean(rand_pvals))

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
                ORGAN_TRANSLATION[organ_dict[organ_time[0]]],
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
def plot_mds_for_tree(
    tree,
    allele_to_cell_state,
    organ_dict,
    out_plot_file,
    min_abund = 20,
    min_cell_types = 5):
    """
    Plot MDS of cell type distances
    """
    similar_cell_type_pairs = [
            ('7B_Eye1', '7B_Eye2')]
    leaves_by_cell_type = {v: [] for v in organ_dict.values()}
    for leaf in tree:
        for cell_type, abund in allele_to_cell_state[leaf.allele_events_list_str].items():
            leaves_by_cell_type[organ_dict[cell_type]].append((leaf, abund))

    cell_type_list = list(organ_dict.values())
    cell_type_list = [
            cell_type for cell_type in cell_type_list
            if sum([m[1] for m in leaves_by_cell_type[cell_type]]) > min_abund]
    if len(cell_type_list) <= min_cell_types:
        print("NOPE")
        return

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
                print(cell_type0, cell_type1, tot_dists/(tot_abund0 * tot_abund1), tot_abund0, tot_abund1)
                X_matrix[idx0, idx1] = tot_dists/(tot_abund0 * tot_abund1)
                X_matrix[idx1, idx0] = tot_dists/(tot_abund0 * tot_abund1)

    mds = MDS(n_components=2, metric=False, max_iter=3000, eps=1e-9, dissimilarity="precomputed")
    noise = np.random.rand(X_matrix.shape[0], X_matrix.shape[1]) * 0.02
    X_matrix += (noise + noise.T)/2
    pos = mds.fit(X_matrix).embedding_
    pyplot.clf()
    fig, ax = pyplot.subplots()
    ax.scatter(pos[:, 0], pos[:, 1], color='turquoise', lw=0, label='MDS')
    for i, cell_type in enumerate(cell_type_list):
        ax.annotate(ORGAN_TRANSLATION[cell_type], (pos[i, 0], pos[i, 1]))

    pyplot.savefig(out_plot_file)
    print("MDS PLOT", out_plot_file)

def plot_mds(
    fitted_bifurc_tree,
    rand_tree,
    allele_to_cell_state,
    organ_dict,
    tot_time,
    out_plot_prefix = "/Users/jeanfeng/Desktop/mds"):
    """
    plot mds with cell types based on MRCA distance
    """
    assign_rand_tree_lengths(rand_tree, tot_time)

    ## MDS for fitted tree -- subtrees
    #for idx, node in enumerate(fitted_bifurc_tree.get_descendants()):
    #    node_dist = node.get_distance(fitted_bifurc_tree)
    #    if len(node.children) == 0:
    #        continue
    #    children_dist = min([child.get_distance(fitted_bifurc_tree) for child in node.children])
    #    if node_dist < 0.15 and children_dist > 0.15:
    #        plot_mds_for_tree(
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
    #    plot_mds_for_tree(
    #        node,
    #        allele_to_cell_state,
    #        organ_dict,
    #        out_plot_file = "%s_rand_%d.png" % (out_plot_prefix, idx))

    # Overall MDS for fitted
    plot_mds_for_tree(
        fitted_bifurc_tree,
        allele_to_cell_state,
        organ_dict,
        out_plot_file = "%s_fitted.png" % out_plot_prefix)

    # Overall MDS for rand tree
    plot_mds_for_tree(
        rand_tree,
        allele_to_cell_state,
        organ_dict,
        out_plot_file = "%s_rand.png" % out_plot_prefix)

def plot_branch_len_time(
    fitted_bifurc_tree,
    rand_tree,
    tot_time,
    out_plot_file,
    num_rands = 5):
    """
    Plot fitted time vs branch length
    """
    X_dist = []
    Y_branch_len = []
    for node in fitted_bifurc_tree.get_descendants():
        if not node.is_leaf():
            node_dist = node.get_distance(fitted_bifurc_tree)
            X_dist.append(node_dist)
            Y_branch_len.append(node.dist)

    if out_plot_file:
        pyplot.clf()
        pyplot.scatter(
                rand_jitter(X_dist, scaling_factor=0.002),
                rand_jitter(Y_branch_len, scaling_factor=0.002),
                s=10)
        pyplot.savefig(out_plot_file)
    print("mle tree", stats.linregress(X_dist, Y_branch_len))

    rand_slopes = []
    rand_corr = []
    rand_pvals = []
    for _ in range(num_rands):
        assign_rand_tree_lengths(rand_tree, tot_time)

        X_dist = []
        Y_branch_len = []
        for node in rand_tree.get_descendants():
            if not node.is_leaf():
                dist = node.get_distance(rand_tree)
                X_dist.append(dist)
                Y_branch_len.append(node.dist)
        slope, _, corr, pval, _ = stats.linregress(X_dist, Y_branch_len)
        rand_slopes.append(slope)
        rand_corr.append(corr)
        rand_pvals.append(pval)
    print("rand tree", np.mean(rand_slopes), np.power(np.mean(rand_corr),2), np.mean(rand_pvals))

"""
plotting my fitted tree now...
"""
def plot_gestalt_tree(
        fitted_bifurc_tree,
        organ_dict,
        allele_to_cell_state,
        cell_state_dict,
        out_plot_file):
    for l in fitted_bifurc_tree:
        allele_str = l.allele_events_list_str
        if l.cell_state is None:
            cell_state_strs = list(allele_to_cell_state[allele_str].keys())
            for c_state_str in cell_state_strs:
                new_child = CellLineageTree(
                    l.allele_list,
                    l.allele_events_list,
                    cell_state_dict[c_state_str],
                    dist = 0,
                    abundance = 1,
                    resolved_multifurcation = True)
                l.add_child(new_child)
    print("num leaves", len(fitted_bifurc_tree))

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
        text = ORGAN_TRANSLATION[organ_key]
        label_dict = {
            "text": text,
            "color": "gray",
            "fontsize": 8}
        legend_colors[color] = label_dict

    print("at plotting phase....")
    plot_tree(
            col_tree,
            out_plot_file,
            width=600,
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
    res = six.moves.cPickle.load(f)["refit"]

with open(rand_tree_file, "rb") as f:
    rand_tree = six.moves.cPickle.load(f)["tree"]

print("plot mds")
plot_mds(
    res.fitted_bifurc_tree,
    rand_tree,
    allele_to_cell_state,
    organ_dict,
    tot_time,
    out_plot_prefix = "/Users/jeanfeng/Desktop/mds")
print("distance to abundance")
plot_distance_to_abundance(
    res.fitted_bifurc_tree,
    rand_tree,
    tot_time,
    out_plot_file = "/Users/jeanfeng/Desktop/scatter_dist_to_abundance.png",
    num_rands = 20)
print("distance to number of descendant cell states")
plot_distance_to_num_cell_states(
    res.fitted_bifurc_tree,
    organ_dict,
    rand_tree,
    tot_time,
    out_plot_file = "/Users/jeanfeng/Desktop/scatter_dist_to_cell_state.png",
    num_rands = 20)
plot_gestalt_tree(
    res.fitted_bifurc_tree,
    organ_dict,
    allele_to_cell_state,
    cell_state_dict,
    "/Users/jeanfeng/Desktop/gestalt_fitted5.png")
print("plot branch length distribution")
plot_branch_len_time(
    res.fitted_bifurc_tree,
    rand_tree,
    tot_time,
    out_plot_file="/Users/jeanfeng/Desktop/scatter_dist_to_branch_len.png",
    num_rands = 20)
