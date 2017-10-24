"""
Compares the true tree to the fitted trees from PHYLIP
"""

import phylip_parse
import pickle

from simulation import BarcodeTree, Barcode, BarcodeForest
from collapsed_tree import CollapsedTree

true_tree_file = "out.pkl"
with open(true_tree_file, "r") as f:
    true_barcode_tree = (pickle.load(f)).trees[0]
print true_barcode_tree.tree
print true_barcode_tree.collapsed_tree.tree

trees = phylip_parse.parse_outfile("outfile")
for t in trees:
    collapsed_est_tree = (CollapsedTree(t, preserve_leaves=True)).tree
    rf_res = collapsed_est_tree.robinson_foulds(
        true_barcode_tree.collapsed_tree.tree, unrooted_trees=True)
    print("RF dist: %f out of %f" % (rf_res[0], rf_res[1]))
