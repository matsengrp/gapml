import six
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from cell_lineage_tree import CellLineageTree
from tree_distance import MRCADistanceMeasurer
import plot_simulation_common

def plot_mrca_matrix(bifurc_tree: CellLineageTree, ref_tree: CellLineageTree, file_name: str, tot_time: float = 1):
    plt.clf()
    if ref_tree is not None:
        mrca_meas = MRCADistanceMeasurer(ref_tree)
        mrca_mat = mrca_meas._get_mrca_matrix(bifurc_tree)
    else:
        mrca_meas = MRCADistanceMeasurer(bifurc_tree)
        mrca_mat = mrca_meas.ref_tree_mrca_matrix
    plt.imshow(mrca_mat, vmin=0, vmax=1)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.tight_layout()
    plt.savefig(file_name)

def plot_tree(tree: CellLineageTree, file_name: str):
    from ete3 import TreeStyle, NodeStyle
    nstyle = NodeStyle()
    nstyle["size"] = 0
    for n in tree.traverse():
        n.set_style(nstyle)

    #offset = np.min([leaf.dist for leaf in tree]) - 0.1
    for leaf_idx, leaf in enumerate(tree):
        leaf.name = "%d:%s" % (leaf_idx, leaf.allele_events_list_str)
        #leaf.dist = leaf.dist - offset
    tree.show_leaf_name = True
    tree.show_branch_length = True
    ts = TreeStyle()
    ts.scale = 100
    tree.render(file_name, w=283, units="mm", tree_style=ts)

#sampling = "same"
#n_bcode = 1
#seed = 1
#file_name = "tmp_mount/simulation_topology_same_diff/_output/model_seed300/%d/%s/double_cut0/true_model.pkl" % (seed, sampling)
#_, true_tree, _ = plot_simulation_common.get_true_model(file_name, None, n_bcode)
#plot_mrca_matrix(true_tree, None, "/Users/jeanfeng/Desktop/true_tree%s%d_mrca.png" % (sampling, seed))
#
#res_file_name = "tmp_mount/simulation_topology_same_diff/_output/model_seed300/%d/%s/double_cut0/num_barcodes1/lambda_known1/abundance_weight0/tune_fittedtree0.pkl" % (seed, sampling)
#_, fitted_tree = plot_simulation_common.get_result(res_file_name)
##plot_tree(fitted_tree, "/Users/jeanfeng/Desktop/fitted_tree%d.png" % sampling)
#plot_mrca_matrix(fitted_tree, true_tree, "/Users/jeanfeng/Desktop/fitted_tree%s%d_mrca.png" % (sampling, seed))


#with open("tmp_mount/simulation_topology_sampling/_output/model_seed3/0/sampling1/true_model.pkl" % n_bcode, "rb") as f:
#    coll_tree = six.moves.cPickle.load(f)
#plot_tree(coll_tree, "/Users/jeanfeng/Desktop/simulation_coll_tree%d.png" % n_bcode)
#plot_mrca_matrix(coll_tree, "/Users/jeanfeng/Desktop/simulation_coll_tree_mrca%d.png" % n_bcode)
#
#with open("tmp_mount/analyze_gestalt/_output/min_abund_5/sum_states_10/extra_steps_0/penalty_params_200/tune_topology_fitted.pkl", "rb") as f:
#    res = six.moves.cPickle.load(f)
#plot_tree(res.fitted_bifurc_tree, "/Users/jeanfeng/Desktop/gestalt_fitted5.png")
#plot_mrca_matrix(res.fitted_bifurc_tree, None, "/Users/jeanfeng/Desktop/gestalt_fitted_mrca5.png")
