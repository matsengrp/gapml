import six
from ete3 import NodeStyle, SeqMotifFace
from plot_mrca_matrices import plot_tree
from matplotlib import pyplot
from scipy import stats
import numpy as np

import collapsed_tree
from cell_lineage_tree import CellLineageTree

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
obs_file = "tmp_mount/analyze_gestalt/_output/min_abund_5/fish_data.pkl"

def get_allele_to_cell_states(obs_dict):
    # Create allele string to cell state
    allele_to_cell_state = {}
    cell_state_dict = {}
    for obs in obs_dict["obs_leaves_by_allele_cell_state"]:
        allele_str_key = CellLineageTree._allele_list_to_str(obs.allele_events_list)
        if allele_str_key in allele_to_cell_state:
            if str(obs.cell_state) not in allele_to_cell_state:
                allele_to_cell_state[allele_str_key].add(str(obs.cell_state))
        else:
            allele_to_cell_state[allele_str_key] = set([str(obs.cell_state)])

        if str(obs.cell_state) not in cell_state_dict:
            cell_state_dict[str(obs.cell_state)] = obs.cell_state

    return allele_to_cell_state, cell_state_dict

def plot_distance_to_abundance(
        fitted_bifurc_tree,
        rand_tree,
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
    for _ in range(num_rands):
        br_scale = 0.8
        has_neg = True
        while has_neg:
            has_neg = False
            for node in rand_tree.traverse():
                if node.is_root():
                    continue
                if node.is_leaf():
                    node.dist = 1 - node.up.get_distance(rand_tree)
                    if node.dist < 0:
                        has_neg = True
                        break
                else:
                    node.dist = np.random.rand() * br_scale
            br_scale *= 0.8

        X_dists = []
        Y_abundance = []
        for node in rand_tree.traverse():
            dist = node.get_distance(rand_tree)
            X_dists.append(dist)
            tot_abundance = sum([leaf.abundance for leaf in node])
            Y_abundance.append(tot_abundance)
        slope, _, corr, _, _ = stats.linregress(X_dists, np.log10(Y_abundance))
        rand_slopes.append(slope)
        rand_corr.append(corr)
    print("rand tree", np.mean(rand_slopes), np.mean(rand_corr))

def plot_distance_to_num_cell_states(
        fitted_bifurc_tree,
        organ_dict,
        rand_tree,
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
                cell_state_strs = allele_to_cell_state[allele_str]
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
    for _ in range(num_rands):
        assign_rand_tree_lengths(rand_tree, TOT_TIME)

        X_dists = []
        Y_n_cell_states = []
        for node in rand_tree.traverse():
            dist = node.get_distance(rand_tree)
            if dist < obs_dict["time"] - 0.01:
                X_dists.append(dist)
                c_state_set = set()
                for leaf in node:
                    allele_str = leaf.allele_events_list_str
                    cell_state_strs = allele_to_cell_state[allele_str]
                    for c_state_str in cell_state_strs:
                        c_state_set.update(c_state_str)
                Y_n_cell_states.append(len(c_state_set))
        slope, _, corr, _, _ = stats.linregress(X_dists, Y_n_cell_states)
        rand_slopes.append(slope)
        rand_corr.append(corr)
    print("rand tree", np.mean(rand_slopes), np.mean(rand_corr))

def plot_majority_cell_appearance_time(
        fitted_bifurc_tree,
        organ_dict,
        out_plot_file = "/Users/jeanfeng/Desktop/cell_appearance_time.png"):
    """
    Plot the time of the progenitor cell of most of a particular cell type
    """
    total_cell_type_nums = {k: 0 for k in organ_dict.keys()}
    raise NotImplementedError()

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
            cell_state_strs = allele_to_cell_state[allele_str]
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
        assert len(l.cell_state) == 1
        nstyle = NodeStyle()
        nstyle["fgcolor"] = ORGAN_COLORS[organ_dict[str(l.cell_state[0])]]
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

bcode_meta = obs_dict["bcode_meta"]
organ_dict = obs_dict["organ_dict"]
print(organ_dict)
1/0

with open(fitted_tree_file, "rb") as f:
    res = six.moves.cPickle.load(f)["refit"]
    print(res.fitted_bifurc_tree.get_ascii(attributes=["abundance"], show_internal=False))

with open(rand_tree_file, "rb") as f:
    rand_tree = six.moves.cPickle.load(f)["tree"]

allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)
print("distance to abundance")
plot_distance_to_abundance(
    res.fitted_bifurc_tree,
    rand_tree,
    out_plot_file = None, #"/Users/jeanfeng/Desktop/scatter_dist_to_abundance.png")
    num_rands = 20)
print("distance to number of descendant cell states")
plot_distance_to_num_cell_states(
    res.fitted_bifurc_tree,
    organ_dict,
    rand_tree,
    out_plot_file = None, #"/Users/jeanfeng/Desktop/scatter_dist_to_cell_state.png")
    num_rands = 20)
#plot_gestalt_tree(
#        res.fitted_bifurc_tree,
#        organ_dict,
#        allele_to_cell_state,
#        cell_state_dict,
#        "/Users/jeanfeng/Desktop/gestalt_fitted5.png")
