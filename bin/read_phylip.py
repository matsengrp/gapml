"""
Render trees from PHYLIP
"""

import phylip_parse
import pickle

from collapsed_tree import CollapsedTree

trees = phylip_parse.parse_outfile("outfile", "event_dict.txt")
for t in trees:
    collapsed_est_tree =  (CollapsedTree(t, preserve_leaves=True)).tree
    rf_res = collapsed_est_tree.robinson_foulds(true_barcode_tree.collapsed_tree.tree, unrooted_trees=True)
    print("RF dist: %f out of %f" % (rf_res[0], rf_res[1]))
