from typing import List, Tuple
import numpy as np

from cell_lineage_tree import CellLineageTree
from barcode_events import Event, BarcodeEvents
from indel_sets import AncState

class AncestralEventsFinder:
    """
    Our in-built engine for finding all possible
    events in the internal nodes.
    """
    def annotate_ancestral_states(self, tree: CellLineageTree):
        """
        Find all possible events in the internal nodes.
        """
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.add_feature("anc_state", AncState.create_for_observed_allele(node.barcode_events))
            elif node.is_root():
                node.add_feature("anc_state", AncState())
                print(node.get_ascii(attributes=["anc_state"], show_internal=True))
            else:
                node_anc_state = self.get_possible_anc_states(node)
                node.add_feature("anc_state", node_anc_state)

    def get_possible_anc_states(self, tree: CellLineageTree):
        # TODO: use a python reduce function?
        children = tree.get_children()
        parent_anc_state = children[0].anc_state
        for c in children[1:]:
            parent_anc_state = AncState.intersect(
                parent_anc_state,
                c.anc_state)
        return parent_anc_state
