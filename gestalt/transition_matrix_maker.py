from typing import Tuple, List, Set, Dict
import itertools

from indel_sets import TargetTract, IndelSet, SingletonWC, TractRepr, Tract
from anc_state import AncState
from cell_lineage_tree import CellLineageTree
from target_status import TargetStatus
from barcode_metadata import BarcodeMetadata
import ancestral_events_finder as anc_evt_finder

from approximator_transition_graph import TransitionToNode, TransitionGraph

class TransitionWrapper:
    def __init__(self, target_statuses: List[TargetStatus], anc_state: AncState, is_leaf: bool):
        self.states = target_statuses
        self.key_dict = {targ_stat: i for i, targ_stat in enumerate(target_statuses)}
        # TODO: rename to num_possible_states
        self.num_likely_states = len(target_statuses)

        self.anc_state = anc_state

        self.is_leaf = is_leaf
        if is_leaf:
            self.leaf_target_status = anc_state.to_max_target_status()

class TransitionMatrixMaker:
    """
    """
    def __init__(self, bcode_metadata: BarcodeMetadata):
        self.bcode_meta = bcode_metadata

    def create_transition_matrix_wrappers(self, model):
        """
        @param model: CLTLikelihoodModel
        """
        # Annotate with ancestral states
        anc_evt_finder.annotate_ancestral_states(model.topology, model.bcode_meta)

        return self.get_target_status_transitions(model.topology)

    def get_target_status_transitions(self, tree: CellLineageTree):
        """
        """
        transition_matrix_states = dict()
        for node in tree.traverse("preorder"):
            wrapper_list = [
                    TransitionWrapper(
                        anc_state.generate_possible_target_statuses(),
                        anc_state,
                        node.is_leaf())
                    for anc_state in node.anc_state_list]
            transition_matrix_states[node.node_id] = wrapper_list
        return transition_matrix_states
