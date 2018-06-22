from typing import List

from anc_state import AncState
from cell_lineage_tree import CellLineageTree
from target_status import TargetStatus
from barcode_metadata import BarcodeMetadata
import ancestral_events_finder as anc_evt_finder

class TransitionWrapper:
    """
    Stores useful information in each node for calculating transition probabilities.
    Most importantly, it stores possible meta-states (i.e. allele states grouped in some way) at this node that can precede the observed data

    In this implementation, we use TargetStatus as a meta-state.
    (see manuscript regarding how exactly the alleles are grouped by target status)
    """
    def __init__(self, target_statuses: List[TargetStatus], anc_state: AncState, is_leaf: bool):
        # The possible states that precede the observed data at the leaves
        # Note that for leaf states, this is the set of all possible states that can precede this leaf,
        # even though the leaf can only be one state.
        self.states = target_statuses
        # The mapping from state to state index number for this node
        self.key_dict = {targ_stat: i for i, targ_stat in enumerate(target_statuses)}
        # TODO: rename to num_possible_states
        self.num_likely_states = len(target_statuses)

        # Ancestral state at this node
        self.anc_state = anc_state

        # If this node is a leaf, then also store the leaf state
        self.leaf_state = None
        if is_leaf:
            self.leaf_state = anc_state.to_max_target_status()

class TransitionWrapperMaker:
    """
    This class helps prune the set of states that we need to calculate transition probabilities for.
    """
    def __init__(self, tree: CellLineageTree, bcode_metadata: BarcodeMetadata):
        self.bcode_meta = bcode_metadata
        self.tree = tree

    def create_transition_wrappers(self):
        """
        @return Dict[node id, List[TransitionWrapperMaker]]: a dictionary that stores the
                list of TransitionWrapperMakers (one for each barcode) for each node
        """
        # Annotate with ancestral states using the efficient upper bounding algo
        anc_evt_finder.annotate_ancestral_states(self.tree, self.bcode_meta)

        # Create a dictionary mapping node to its TransitionWrapper
        transition_matrix_states = dict()
        for node in self.tree.traverse("preorder"):
            wrapper_list = [
                    TransitionWrapper(
                        anc_state.generate_possible_target_statuses(),
                        anc_state,
                        node.is_leaf())
                    for anc_state in node.anc_state_list]
            transition_matrix_states[node.node_id] = wrapper_list
        return transition_matrix_states
