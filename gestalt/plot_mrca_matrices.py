import six
import scipy.stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ete3 import TreeStyle, NodeStyle

from cell_lineage_tree import CellLineageTree
from tree_distance import MRCADistanceMeasurer

def plot_mrca_matrix(bifurc_tree: CellLineageTree, file_name: str):
    mrca_meas = MRCADistanceMeasurer(bifurc_tree)
    mrca_mat = mrca_meas.ref_tree_mrca_matrix
    plt.imshow(mrca_mat)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.tight_layout()
    plt.savefig(file_name)

def plot_tree(tree: CellLineageTree, file_name: str):
    nstyle = NodeStyle()
    nstyle["size"] = 0
    for n in tree.traverse():
        n.set_style(nstyle)

    for leaf_idx, leaf in enumerate(tree):
        leaf.name = "%d:%s" % (leaf_idx, leaf.allele_events_list_str)
    tree.show_leaf_name = True
    tree.show_branch_length = True
    ts = TreeStyle()
    ts.scale = 100
    tree.render(file_name, w=183, units="mm", tree_style=ts)

# Save the data
with open("tmp_mount/_output_worm2/parsimony_tree1_fitted.pkl", "rb") as f:
    res = six.moves.cPickle.load(f)
plot_tree(res.fitted_bifurc_tree, "/Users/jeanfeng/Desktop/parsimony_tree1_fitted.png")
