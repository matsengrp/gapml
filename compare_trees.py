"""
Compares the true tree to the fitted trees from PHYLIP
"""

import phylip_parse
import pickle

from simulation import BarcodeTree, Barcode, BarcodeForest

true_tree_file = "out.pkl"
with open(true_tree_file, "r") as f:
    true_tree = (pickle.load(f)).trees[0]

trees = phylip_parse.parse_outfile("outfile")
for t in trees:
    rf_res = t.robinson_foulds(true_tree.tree)
    print("%f out of %f" % (rf_res[0], rf_res[1]))
