import networkx as nx
import seaborn as sns
import numpy as np
import six
from cell_lineage_tree import CellLineageTree
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree, NodeStyle

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

FISH = "ADR1"
if FISH == "ADR1":
    fitted_tree_file = "_output/gestalt_aws/ADR1_fitted.pkl"
    obs_file = "_output/gestalt_aws/ADR1_fish_data.pkl"
elif FISH == "ADR2":
    fitted_tree_file = "tmp_mount/analyze_gestalt/_output/ADR2_abund1/sum_states_10/extra_steps_0/tune_pen_hanging.pkl"
    obs_file = "tmp_mount/analyze_gestalt/_output/ADR2_abund1/fish_data_restrict_with_cell_types.pkl"

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
    "Heart_GFP+": 7,
    "Heart_chunk": 8,
    "Heart_GFP-": 9,
    "Heart_diss": 10,
    #"Heart_chunk": 7,
    #"Heart_GFP-": 7,
    #"Heart_diss": 7,
}
ORGAN_LABELS = [
    "Brain",
    "Eye1",
    "Eye2",
    "Gills",
    "Intestine",
    "Upper_GI",
    "Blood",
#    "Heart"]
    "Heart_GFP+",
    "Heart_chunk",
    "Heart_GFP-",
    "Heart_diss",
]
NUM_ORGANS = len(ORGAN_LABELS)

with open(fitted_tree_file, "rb") as f:
    if FISH == "ADR1":
        fitted_bifurc_tree = six.moves.cPickle.load(f)[0]["best_res"].fitted_bifurc_tree
    else:
        res = six.moves.cPickle.load(f)["final_fit"]
with open(obs_file, "rb") as f:
    obs_dict = six.moves.cPickle.load(f)

tot_time = obs_dict["time"]
bcode_meta = obs_dict["bcode_meta"]
organ_dict = obs_dict["organ_dict"]
allele_to_cell_state, cell_state_dict = get_allele_to_cell_states(obs_dict)

fitted_bifurc_tree.label_dist_to_roots()
for node in fitted_bifurc_tree.traverse('postorder'):
    dist = node.get_distance(fitted_bifurc_tree)
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
tot_cell_type_abundances = np.zeros(NUM_ORGANS)
for node in fitted_bifurc_tree:
    if not node.is_leaf() and len(node.cell_types) > 1:
        continue
    node_cell_type = list(node.cell_types)[0]
    tot_cell_type_abundances[node_cell_type] += node.abundance

    observed_cell_types = set([node_cell_type])
    up_node = node.up
    while not up_node.is_root() and len(observed_cell_types) < NUM_ORGANS:
        diff_cell_types = up_node.cell_types - observed_cell_types
        for up_cell_type in diff_cell_types:
            X_matrix[node_cell_type, up_cell_type] += (1 - up_node.dist_to_root) * node.abundance
        observed_cell_types.update(diff_cell_types)
        up_node = up_node.up

for i in range(NUM_ORGANS):
    print(tot_cell_type_abundances[i])
    print(ORGAN_LABELS[i])
    X_matrix[i,:] = X_matrix[i,:]/tot_cell_type_abundances[i]
print(X_matrix)
sym_X_matrix = (X_matrix+ X_matrix.T)/2

# HEATMAP
mask = np.zeros(X_matrix.shape, dtype=bool)
for i in range(NUM_ORGANS):
    if tot_cell_type_abundances[i] < 50:
        mask[i,:] = True
        mask[:,i] = True
plt.clf()
sns.heatmap(sym_X_matrix,
        xticklabels=ORGAN_LABELS,
        yticklabels=ORGAN_LABELS,
        mask=mask,
        annot=True)
out_plot_file = "_output/sym_heat_%s.png" % FISH
plt.savefig(out_plot_file)
print("matrix PLOT", out_plot_file)

# TRY NEIGHBOR JOIN
#dm = DistanceMatrix(sym_X_matrix, ORGAN_LABELS)
#tree = nj(dm, disallow_negative_branch_length=True)
#print(tree.ascii_art())
#out_newick = "_output/newick.txt" 
#with open(out_newick, "w") as f:
#    tree.write(f)
#
#t = Tree(out_newick)
#print(t.get_ascii(attributes=["dist"]))
#nstyle = NodeStyle()
#nstyle["size"] = 0
#for n in t.traverse():
#    if not n.is_leaf():
#        n.set_style(nstyle)
#t.render("_output/meta_cell_tree.png", w=183, units="mm")
#
## DO MDS
#mds = MDS(n_components=2, metric=False, max_iter=3000, eps=1e-9, dissimilarity="precomputed")
#pos = mds.fit(sym_X_matrix).embedding_
#plt.clf()
#fig, ax = plt.subplots()
#ax.scatter(pos[:, 0], pos[:, 1], lw=0, label='MDS')
#for i, org_label in enumerate(ORGAN_LABELS):
#    ax.annotate(org_label, (pos[i, 0], pos[i, 1]))
#out_plot_file = "_output/mds.png"
#plt.savefig(out_plot_file)
