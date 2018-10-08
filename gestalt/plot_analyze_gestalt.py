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
#print("distance to abundance")
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
