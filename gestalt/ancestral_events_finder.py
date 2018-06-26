import logging
from typing import List, Tuple
import numpy as np

from cell_lineage_tree import CellLineageTree
from allele_events import Event, AlleleEvents
from anc_state import AncState

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
            node.add_feature("anc_state_list", [
                AncState.create_for_observed_allele(evts, bcode_meta)
                for evts in node.allele_events_list])
        elif node.is_root():
            node.add_feature("anc_state_list", [AncState() for _ in range(bcode_meta.num_barcodes)])
        else:
            node_anc_state = get_possible_anc_states(node)
            node.add_feature("anc_state_list", node_anc_state)
        node.add_feature("anc_state_list_str", [str(k) for k in node.anc_state_list])
    logging.info("Ancestral state")
    logging.info(node.get_ascii(attributes=["anc_state_list_str"]))

def get_possible_anc_states(tree: CellLineageTree):
    children = tree.get_children()
    parent_anc_state_list = []
    for bcode_idx, par_anc_state in enumerate(children[0].anc_state_list):
        for c in children[1:]:
            par_anc_state = AncState.intersect(
                par_anc_state,
                c.anc_state_list[bcode_idx])
        parent_anc_state_list.append(par_anc_state)
    return parent_anc_state_list
