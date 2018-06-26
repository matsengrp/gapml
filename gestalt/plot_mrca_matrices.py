import six
import scipy.stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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
