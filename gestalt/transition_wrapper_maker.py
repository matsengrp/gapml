from typing import List, Dict, Set
import random
import logging
import numpy as np

from anc_state import AncState
from cell_lineage_tree import CellLineageTree
from target_status import TargetStatus
from indel_sets import TargetTract, SingletonWC
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
        self.num_possible_states = len(target_statuses)

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
    def __init__(
            self,
            tree: CellLineageTree,
            bcode_metadata: BarcodeMetadata,
            max_extra_steps: int = 1,
            max_num_states: int = None):
        """
        @param tree: the tree to create transition wrappers for
        @param max_extra_steps: number of extra steps to search for possible ancestral states
        @param max_num_states: maximum number of possible ancestral states befor we decide to use an approximation instead
        """
        self.bcode_meta = bcode_metadata
        self.tree = tree

        self.max_extra_steps = max_extra_steps
        # Set the max number of states to the thing specified or make it super big so we will
        # just marginalize over everything
        self.max_num_states = max_num_states if max_num_states is not None else np.power(
                2, bcode_metadata.n_targets)

    def create_transition_wrappers(self):
        """
        @return Dict[node id, List[TransitionWrapperMaker]]: a dictionary that stores the
                list of TransitionWrapperMakers (one for each barcode) for each node
        """
        targ_stat_transitions_dict, targ_stat_inv_transitions_dict = TargetStatus.get_all_transitions(
                self.bcode_meta)
        # Annotate with ancestral states using the efficient upper bounding algo
        anc_evt_finder.annotate_ancestral_states(self.tree, self.bcode_meta)

        # Create a dictionary mapping node to its TransitionWrapper
        transition_matrix_states = dict()
        for node in self.tree.traverse("preorder"):
            wrapper_list = []
            for idx, anc_state in enumerate(node.anc_state_list):
                possible_targ_statuses = anc_state.generate_possible_target_statuses()
                num_possible_states = len(possible_targ_statuses)
                only_sgwcs = [evt.__class__ == SingletonWC for evt in node.allele_events_list[idx].events]
                if only_sgwcs and num_possible_states > self.max_num_states:
                    # Subset the possible internal states as lower bound
                    # The subset is going to be those within a small number of steps
                    # We will only subset for ancestral states corresponding to singleton wcs since we are
                    # relatively confident that the most likely ancestral states have the singleton wc
                    # at this internal node and it is most likely the singleton wcs were reached via a small
                    # number of steps
                    parent_statuses = transition_matrix_states[node.up.node_id][idx].states
                    max_target_status = anc_state.to_max_target_status()

                    close_states = self.get_states_close_by(
                            parent_statuses,
                            targ_stat_transitions_dict,
                            targ_stat_inv_transitions_dict,
                            max_target_status)

                    target_statuses = list(
                            set(parent_statuses) | close_states | set([anc_state.to_max_target_status()]))
                    logging.info(
                        "Subsampling states for node %d, parent # states %d, node # states %d, reduced to # states %d",
                        node.node_id,
                        len(parent_statuses),
                        num_possible_states,
                        len(target_statuses))
                    assert num_possible_states >= len(target_statuses)
                else:
                    target_statuses = anc_state.generate_possible_target_statuses()

                if len(target_statuses) > self.max_num_states:
                    logging.info(
                        "WARNING: (too many?) possible states at this ancestral node: %d states",
                        len(target_statuses))

                wrapper_list.append(TransitionWrapper(
                    target_statuses,
                    anc_state,
                    node.is_leaf()))

            transition_matrix_states[node.node_id] = wrapper_list
        return transition_matrix_states

    def get_states_close_by(
            self,
            parent_statuses: List[TargetStatus],
            targ_stat_transitions_dict: Dict[TargetStatus, Dict[TargetStatus, List[TargetTract]]],
            targ_stat_inv_transitions_dict: Dict[TargetStatus, Set[TargetStatus]],
            end_target_status: TargetStatus):
        """
        @param parent_statuses: the parent target statuses
        @param targ_stat_transitions_dict: dictionary specifying all possible transitions between target statuses
        @param targ_stat_inv_transitions_dict: dictionary specifying all possible inverse/backwards transitions between target statuses
        @param end_target_status: the ending target status

        @return Set[TargetStatus] -- target statuses that are within `self.max_extra_steps` steps away from `end_target_status`
                when starting at any state in `parent_statuses`.
        """
        # label all nodes starting from the parent statuses with their closest distance (up to the max distance)
        dist_to_start_dict = {p: 0 for p in parent_statuses}
        states_to_explore = set(parent_statuses)
        for i in range(self.max_extra_steps):
            new_states = set()
            for state in states_to_explore:
                one_step_states = set(list(targ_stat_transitions_dict[state].keys()))
                for c in one_step_states:
                    if c not in dist_to_start_dict:
                        dist_to_start_dict[c] = i + 1
                new_states.update(one_step_states)
            states_to_expore = new_states

        # label all nodes starting from the end target status with their closest distance (up to the max distance)
        dist_to_max_dict = {end_target_status: 0}
        states_to_explore = set(end_target_status)
        for i in range(self.max_extra_steps):
            new_states = set()
            for state in states_to_explore:
                if state in targ_stat_inv_transitions_dict:
                    one_step_inv_states = targ_stat_inv_transitions_dict[state]
                    for c in one_step_inv_states:
                        if c not in dist_to_start_dict:
                            dist_to_max_dict[c] = i + 1
                    new_states.update(one_step_states)
            states_to_expore = new_states

        # Now just take the nodes that have the sum of the "from" and "to" distance to be no more than the max distance
        good_states = set()
        for state, dist_to_start in dist_to_start_dict.items():
            if state in dist_to_max_dict:
                if dist_to_max_dict[state] + dist_to_start <= self.max_extra_steps + 1:
                    good_states.add(state)
        return good_states
