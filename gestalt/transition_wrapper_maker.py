from typing import List, Dict, Set
import random
import logging
import numpy as np
import time
from queue import Queue, PriorityQueue

from anc_state import AncState
from cell_lineage_tree import CellLineageTree
from target_status import TargetStatus
from indel_sets import TargetTract, TargetTractTuple, SingletonWC
from barcode_metadata import BarcodeMetadata
import ancestral_events_finder as anc_evt_finder

"""
For each node, track the possible TARGET TRACTS as well as translating that into the possible target statuses.
We restrict the possible states by restricting in the space of target tracts. This gets translated to the possible target statuses.
This is the only way we can properly count "number of steps" between possible barcode states.
Otherwise we might be counting the wrong sequence of steps that are impossible for the barcode.
"""

class TransitionWrapper:
    """
    Stores useful information in each node for calculating transition probabilities.
    Most importantly, it stores possible meta-states (i.e. allele states grouped in some way) at this node that can precede the observed data

    In this implementation, we use TargetStatus as a meta-state.
    (see manuscript regarding how exactly the alleles are grouped by target status)
    """
    def __init__(
            self,
            target_tract_tuples: List[TargetTractTuple],
            anc_state: AncState,
            is_leaf: bool):
        self.target_tract_tuples = target_tract_tuples
        # The possible states that precede the observed data at the leaves
        # Note that for leaf states, this is the set of all possible states that can precede this leaf,
        # even though the leaf can only be one state.
        target_statuses = list(set([
            TargetStatus.from_target_tract_tuple(tt_tuple)
            for tt_tuple in target_tract_tuples]))
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
            assert self.leaf_state in target_statuses

class TransitionWrapperMaker:
    """
    This class helps prune the set of states that we need to calculate transition probabilities for.
    """
    def __init__(
            self,
            tree: CellLineageTree,
            bcode_metadata: BarcodeMetadata,
            max_extra_steps: int = 1,
            max_sum_states: int = 3000):
        """
        @param tree: the tree to create transition wrappers for
        @param max_extra_steps: number of extra steps to search for possible ancestral states
        """
        self.bcode_meta = bcode_metadata
        self.tree = tree

        self.max_extra_steps = max_extra_steps
        self.max_sum_states = max_sum_states

    def _get_close_transition_wrapper(
            self,
            node: CellLineageTree,
            idx: int,
            max_parsimony_sgs: Dict[int, List],
            parent_tt_tuples: List[TargetTractTuple]):
        anc_state = node.anc_state_list[idx]
        maximal_sgs = max_parsimony_sgs[node.node_id][idx]
        maximal_up_sgs = max_parsimony_sgs[node.up.node_id][idx]
        min_required_steps = len(set(maximal_sgs) - set(maximal_up_sgs))
        requested_target_status = TargetStatus.from_target_tract_tuple(TargetTractTuple(*[sg.get_target_tract() for sg in maximal_sgs]))

        states_too_many = True
        max_extra_steps = self.max_extra_steps
        if self.max_sum_states is not None:
            max_extra_steps = self.max_extra_steps if np.power(2, min_required_steps) <= self.max_sum_states else max(self.max_extra_steps - 1, 0)
        while states_too_many and max_extra_steps >= 0:
            close_target_tract_tuples = self.get_states_close_by(
                    min_required_steps + max_extra_steps,
                    min_required_steps,
                    parent_tt_tuples,
                    anc_state,
                    requested_target_status)
            transition_wrap = TransitionWrapper(
                close_target_tract_tuples,
                anc_state,
                node.is_leaf())
            states_too_many = self.max_sum_states is not None and len(transition_wrap.states) > self.max_sum_states
            if states_too_many:
                max_extra_steps -= 1
        return transition_wrap

    def create_transition_wrappers(self):
        """
        @return Dict[node id, List[TransitionWrapperMaker]]: a dictionary that stores the
                list of TransitionWrapperMakers (one for each barcode) for each node
        """
        # Annotate with ancestral states using the efficient upper bounding algo
        anc_evt_finder.annotate_ancestral_states(self.tree, self.bcode_meta)
        max_parsimony_sgs = anc_evt_finder.get_max_parsimony_anc_singletons(self.tree, self.bcode_meta)

        # Create a dictionary mapping node to its TransitionWrapper
        transition_matrix_states = dict()
        for node in self.tree.traverse("preorder"):
            transition_matrix_states[node.node_id] = []
        transition_matrix_states[self.tree.node_id] = [TransitionWrapper(
                        [TargetTractTuple()],
                        anc_state,
                        self.tree.is_leaf()) for anc_state in self.tree.anc_state_list]

        for up_node in self.tree.traverse("preorder"):
            if up_node.is_leaf():
                continue
            for idx, up_anc_state in enumerate(up_node.anc_state_list):
                # List out possible internal states but only consider those within a small number of steps
                # of the possible max parsimony ancestral states
                # We do this by counting those wwithin the minimal target status of all possible max parsimony
                # ancestral states
                parent_target_tract_tuples = transition_matrix_states[up_node.node_id][idx].target_tract_tuples
                filtered_down_tt_tuples = set(parent_target_tract_tuples)
                for node in up_node.children:
                    transition_wrap = self._get_close_transition_wrapper(
                            node,
                            idx,
                            max_parsimony_sgs,
                            parent_target_tract_tuples)
                    filtered_down_tt_tuples = filtered_down_tt_tuples.intersection(set(transition_wrap.target_tract_tuples))

                for node in up_node.children:
                    transition_wrap = self._get_close_transition_wrapper(
                            node,
                            idx,
                            max_parsimony_sgs,
                            filtered_down_tt_tuples)
                    transition_matrix_states[node.node_id].append(transition_wrap)
                    logging.info(
                        "Subsampling states for node %d, parent # states %d, node # subsampled states %d, (node target tract tuples %d)",
                        node.node_id,
                        len(transition_matrix_states[node.up.node_id][idx].states),
                        len(transition_wrap.states),
                        len(transition_wrap.target_tract_tuples))

        return transition_matrix_states

    def get_states_close_by(
            self,
            max_steps: int,
            min_steps_to_sg: int,
            parent_target_tract_tuples: List[TargetTractTuple],
            anc_state: AncState,
            requested_target_status: TargetStatus):
        """
        @param parent_statuses: the parent target statuses
        @param targ_stat_transitions_dict: dictionary specifying all possible transitions between target statuses
        @param targ_stat_inv_transitions_dict: dictionary specifying all possible inverse/backwards transitions between target statuses
        @param end_target_status: the ending target status
        @param anc_state: AncState for specifying the possible ancestral states of this node
        @param requred_target_status: we want target tract tuples that are close to this target status

        @return List[TargetTractTuples] -- target tract tuples within max_steps of
        """
        if max_steps == 0:
            # special case: there are no steps we are allowed to take.
            # then the only possible state is exactly the state of the node
            # this should really only happen for nodes with no events or leaf nodes
            assert all([indel_set.__class__ == SingletonWC for indel_set in anc_state.indel_set_list])
            no_step_max_state = [sg.get_target_tract() for sg in anc_state.get_singleton_wcs()]
            return [TargetTractTuple(*no_step_max_state)]

        # Pick out states along paths that reach the minimal target status
        # of all possible max parsimony ancestral states within the specified `max_steps`
        sg_max_inactive_targs = requested_target_status.get_inactive_targets(self.bcode_meta)
        # Maps distance to nodes that we can reach in exactly that distance
        state_to_parents_dict = {}
        state_queue = PriorityQueue()
        for p in parent_target_tract_tuples:
            parent_targ_stat = TargetStatus.from_target_tract_tuple(p)
            parent_inactive_targs = parent_targ_stat.get_inactive_targets(self.bcode_meta)
            parent_passed_requested = set(sg_max_inactive_targs) <= set(parent_inactive_targs)
            parent_state = (p, parent_passed_requested)
            state_queue.put((0, parent_state))
            state_to_parents_dict[parent_state] = {}
            state_to_parents_dict[parent_state][0] = set()

        max_targets = set(anc_state.to_max_target_status().get_inactive_targets(self.bcode_meta))
        # Find all paths of at most max_steps long where paths are all possible according to `anc_state`
        while not state_queue.empty():
            dist, (state, state_passed_requested) = state_queue.get_nowait()
            if dist >= max_steps:
                continue

            targ_stat = TargetStatus.from_target_tract_tuple(state)
            active_any_targs = list(sorted(list(
                max_targets - set(targ_stat.get_inactive_targets(self.bcode_meta)))))
            possible_targ_tracts = targ_stat.get_possible_target_tracts(
                    self.bcode_meta,
                    active_any_targs=active_any_targs)

            for possible_tt in possible_targ_tracts:
                new_state = TargetTractTuple.merge([state, (possible_tt,)])
                if not anc_state.is_possible(new_state):
                    continue

                new_state_passed_requested = state_passed_requested
                if not state_passed_requested:
                    new_state_targ_stat = TargetStatus.from_target_tract_tuple(new_state)
                    new_state_inactive_targs = new_state_targ_stat.get_inactive_targets(self.bcode_meta)
                    new_state_passed_requested = set(sg_max_inactive_targs) <= set(new_state_inactive_targs)
                child_state = (new_state, new_state_passed_requested)
                state_queue.put((dist + 1, child_state))

                if child_state not in state_to_parents_dict:
                    state_to_parents_dict[child_state] = {}
                if dist + 1 not in state_to_parents_dict[child_state]:
                    state_to_parents_dict[child_state][dist + 1] = set()
                state_to_parents_dict[child_state][dist + 1].add(
                        (state, state_passed_requested))

        # Backtracking to get the states along paths of less than max_steps that passed thru a requested state
        close_states = set()
        for state, state_to_par_step_ct_dict in state_to_parents_dict.items():
            passed_requested = state[1]
            if not passed_requested:
                continue

            for step_count in state_to_par_step_ct_dict.keys():
                ancestor_queue = Queue()
                ancestor_queue.put((state, step_count))
                while not ancestor_queue.empty():
                    ancestor, back_count = ancestor_queue.get_nowait()
                    if back_count >= 1:
                        if (ancestor, back_count) not in close_states and back_count in state_to_parents_dict[ancestor]:
                            for anc_parent in state_to_parents_dict[ancestor][back_count]:
                                ancestor_queue.put((anc_parent, back_count - 1))
                    close_states.add((ancestor, back_count))

        assert len(close_states) > 0
        return list(set([state for (state, _), _ in close_states]))
