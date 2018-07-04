from typing import List, Dict
import random
import logging
import numpy as np

from anc_state import AncState
from cell_lineage_tree import CellLineageTree
from target_status import TargetStatus
from indel_sets import TargetTract
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
        targ_stat_transitions_dict = TargetStatus.get_all_transitions(self.bcode_meta)
        # Annotate with ancestral states using the efficient upper bounding algo
        anc_evt_finder.annotate_ancestral_states(self.tree, self.bcode_meta)

        # Create a dictionary mapping node to its TransitionWrapper
        transition_matrix_states = dict()
        for node in self.tree.traverse("preorder"):
            wrapper_list = []
            for idx, anc_state in enumerate(node.anc_state_list):
                possible_targ_statuses = anc_state.generate_possible_target_statuses()
                num_possible_states = len(possible_targ_statuses)
                if node.is_leaf() and num_possible_states > self.max_num_states:
                    # Subset the possible internal states as lower bound
                    # We only do this for leaves since this is easy to deal with
                    # (subsetting internal node ancestral states seems tricky -- the approx error propagates)
                    # The subset is going to be those within a small number of steps
                    parent_statuses = transition_matrix_states[node.up.node_id][idx].states

                    close_states = self.get_states_close_by(
                            possible_targ_statuses,
                            parent_statuses,
                            targ_stat_transitions_dict)

                    target_statuses = list(
                            set(parent_statuses) | close_states | set([anc_state.to_max_target_status()]))
                    logging.info(
                        "Subsampling states for node %d, parent # states %d, node # states %d, reduced to # states %d",
                        node.node_id,
                        len(parent_statuses),
                        num_possible_states,
                        len(target_statuses))
                else:
                    target_statuses = anc_state.generate_possible_target_statuses()

                if len(target_statuses) > self.max_num_states:
                    logging.info(
                        "WARNING: (too many?) possible states at this ancestral node: %d states",
                        num_possible_states)

                wrapper_list.append(TransitionWrapper(
                    target_statuses,
                    anc_state,
                    node.is_leaf()))

            transition_matrix_states[node.node_id] = wrapper_list
        return transition_matrix_states

    def get_states_close_by(
            self,
            possible_targ_statuses: List[TargetStatus],
            parent_statuses: List[TargetStatus],
            targ_stat_transitions_dict: Dict[TargetStatus, Dict[TargetStatus, List[TargetTract]]]):
        """
        @param possible_targ_statuses: possible target statuses
        @param parent_statuses: the parent target statuses
        @param targ_stat_transitions_dict: dictionary specifying all possible transitions between target statuses

        @return List[TargetStatus] -- a subset of `possible_states` that are within `self.max_extra_steps`
                steps away from a state in `parent_statuses`.
                Code performs a BFS search essentially.
        """
        possible_states = set(possible_targ_statuses)

        # store the previously seen states that we no longer need to explore again
        prev_seen_states = set(parent_statuses)
        # store the states we need to explore still
        states_to_expand = set(parent_statuses)
        for i in range(self.max_extra_steps):
            next_step_states = set()

            for state in states_to_expand:
                one_step_states = set(list(targ_stat_transitions_dict[state].keys()))
                # The states to look at next must be one step away, is a possible ancestral state,
                # and not previously seen
                steps_to_expand = (one_step_states & possible_states) - prev_seen_states
                next_step_states.update(steps_to_expand)

            # update the previously seen states and the states we still need to explore
            states_to_expand = next_step_states
            prev_seen_states.update(states_to_expand)

        return prev_seen_states
