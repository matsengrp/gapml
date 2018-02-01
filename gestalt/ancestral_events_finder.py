from typing import List, Tuple
import numpy as np

from cell_lineage_tree import CellLineageTree
from allele_events import Event, AlleleEvents
from indel_sets import AncState

from barcode_metadata import BarcodeMetadata

"""
Our in-built engine for finding all possible
events in the internal nodes.
"""
def annotate_ancestral_states(tree: CellLineageTree, bcode_meta: BarcodeMetadata):
    """
    Find all possible events in the internal nodes.
    """
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.add_feature("anc_state", AncState.create_for_observed_allele(node.allele_events, bcode_meta))
        elif node.is_root():
            node.add_feature("anc_state", AncState())
        else:
            node_anc_state = get_possible_anc_states(node)
            node.add_feature("anc_state", node_anc_state)

def get_possible_anc_states(tree: CellLineageTree):
    # TODO: use a python reduce function?
    children = tree.get_children()
    parent_anc_state = children[0].anc_state
    for c in children[1:]:
        parent_anc_state = AncState.intersect(
            parent_anc_state,
            c.anc_state)
    return parent_anc_state
