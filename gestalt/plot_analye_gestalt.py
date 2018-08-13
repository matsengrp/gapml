import six
from ete3 import NodeStyle, SeqMotifFace
from plot_mrca_matrices import plot_tree
from matplotlib import pyplot
from scipy import stats
import numpy as np

import collapsed_tree
from cell_lineage_tree import CellLineageTree

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

fitted_tree_file = "tmp_mount/analyze_gestalt/_output/min_abund_5/sum_states_10/extra_steps_0/penalty_params_200/tune_topology_fitted.pkl"
obs_file = "tmp_mount/analyze_gestalt/_output/min_abund_5/fish_data.pkl"

with open(obs_file, "rb") as f:
    obs_dict = six.moves.cPickle.load(f)

bcode_meta = obs_dict["bcode_meta"]
organ_dict = obs_dict["organ_dict"]


# Create allele string to cell state
tot_uniq_cell_state_allele_pairs = 0
allele_to_cell_state = {}
for obs in obs_dict["obs_leaves"]:
    allele_str_key = CellLineageTree._allele_list_to_str(obs.allele_events_list)
    cell_states = obs.cell_state
    idx_to_cell_state = {str(c_state): c_state for c_state in cell_states}
    count_dict = {}
    for c_state in cell_states:
        if str(c_state) in count_dict:
            count_dict[str(c_state)] += 1
        else:
            count_dict[str(c_state)] = 1
    cell_state_list = []
    curr_thres = THRES
    while len(cell_state_list) == 0:
        cell_state_list = [
            idx_to_cell_state[c_state_str]
            for c_state_str, count in count_dict.items()
            if count > curr_thres]
        curr_thres -= 1
    allele_to_cell_state[allele_str_key] = cell_state_list
    tot_uniq_cell_state_allele_pairs += len(allele_to_cell_state[allele_str_key])

print(organ_dict)
print("Number of observed alleles", tot_uniq_cell_state_allele_pairs)

with open(fitted_tree_file, "rb") as f:
    res = six.moves.cPickle.load(f)

for l in res.fitted_bifurc_tree:
    allele_str = l.allele_events_list_str
    if l.cell_state is None:
        l.cell_state = allele_to_cell_state[allele_str]
        assert len(l.cell_state) > 0
        for c_state in l.cell_state:
            new_child = CellLineageTree(
                l.allele_list,
                l.allele_events_list,
                [c_state],
                dist = 0,
                abundance = 1,
                resolved_multifurcation = True)
            l.add_child(new_child)
print("num leaves", len(res.fitted_bifurc_tree))

for l in res.fitted_bifurc_tree:
    assert len(l.cell_state) == 1
    nstyle = NodeStyle()
    nstyle["fgcolor"] = ORGAN_COLORS[organ_dict[str(l.cell_state[0])]]
    nstyle["size"] = 10 
    l.set_style(nstyle)

for leaf in res.fitted_bifurc_tree:
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
for node in res.fitted_bifurc_tree.get_descendants():
    if node.dist < COLLAPSE_DIST:
        node.dist = 0
col_tree = collapsed_tree.collapse_zero_lens(res.fitted_bifurc_tree)

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
        col_tree, # res.fitted_bifruc_tree
        "/Users/jeanfeng/Desktop/gestalt_fitted5.png",
        width=300,
        show_leaf_name=False,
        legend_colors=legend_colors)

"""
Understand if distance to root is inversely related to number of
different cell states in leaves
"""
def rand_jitter(arr, scaling_factor=0.003):
    stdev = scaling_factor*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

X_dists = []
Y_n_cell_states = []
scatter_X_dists = []
scatter_Y_n_cell_states = []
colors = []
for node in res.fitted_bifurc_tree.traverse():
    dist = node.get_distance(res.fitted_bifurc_tree)
    if dist < obs_dict["time"] - 0.01:
        X_dists.append(dist)
        c_state_set = set()
        for leaf in node:
            for c_state in leaf.cell_state:
                c_state_set.add(str(c_state))
                colors.append(
                        ORGAN_COLORS[organ_dict[str(c_state)]])

        n_cell_states = len(c_state_set)
        Y_n_cell_states.append(n_cell_states)
        jitter = (np.random.rand() - 0.5) * 0.1
        scatter_X_dists += [dist + jitter] * n_cell_states
        jitter = (np.random.rand() - 0.5) * 0.5
        scatter_Y_n_cell_states += [n_cell_states + jitter] * n_cell_states

pyplot.clf()
pyplot.scatter(
        rand_jitter(scatter_X_dists, scaling_factor=0.002),
        rand_jitter(scatter_Y_n_cell_states, scaling_factor=0.002),
        c=colors,
        alpha=0.25,
        marker="o",
        s=10)
pyplot.savefig("/Users/jeanfeng/Desktop/scatter_dist_to_cell_state.png")
print(stats.linregress(X_dists, Y_n_cell_states))
