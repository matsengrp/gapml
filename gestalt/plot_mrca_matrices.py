import six
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from typing import Dict

from cell_lineage_tree import CellLineageTree
from tree_distance import MRCADistanceMeasurer
import plot_simulation_common

COLORS = [
        "Orange",
        "GreenYellow",
        "Green",
        "Maroon",
        "Brown",
        "LightPink",
        "Red",
        "Cyan",
        "Indigo"]

def plot_mrca_matrix(bifurc_tree: CellLineageTree, ref_tree: CellLineageTree, file_name: str, tot_time: float = 1):
    plt.clf()
    if ref_tree is not None:
        mrca_meas = MRCADistanceMeasurer(ref_tree)
        mrca_mat = mrca_meas._get_mrca_matrix(bifurc_tree)
    else:
        mrca_meas = MRCADistanceMeasurer(bifurc_tree)
        mrca_mat = mrca_meas.ref_tree_mrca_matrix
    plt.imshow(mrca_mat, vmin=0, vmax=tot_time)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.tight_layout()
    plt.savefig(file_name)

def plot_tree(
        tree: CellLineageTree,
        file_name: str,
        width: int=300,
        height: int=None,
        show_leaf_name: bool = True,
        legend_colors: Dict = {}):
    from ete3 import CircleFace, TreeStyle, NodeStyle, RectFace
    print(file_name)

    nstyle = NodeStyle()
    nstyle["size"] = 0
    for n in tree.traverse():
        if not n.is_leaf():
            n.set_style(nstyle)

    if show_leaf_name:
        #for leaf_idx, leaf in enumerate(tree):
        #    leaf.name = "%d:%s" % (leaf_idx, leaf.allele_events_list_str)
        tree.show_leaf_name = show_leaf_name

    tree.show_branch_length = True
    ts = TreeStyle()
    ts.scale = 100
    ts.min_leaf_separation = 2

    # Add legend to top of the plot
    for color, label_dict in legend_colors.items():
        ts.legend.add_face(
                RectFace(
                    50 * label_dict["fontsize"],
                    height=2 * label_dict["fontsize"],
                    fgcolor="black",
                    bgcolor=color,
                    label=label_dict),
                column=0)
    tree.render(file_name, w=width, h=height, units="mm", tree_style=ts)
    print("done")
