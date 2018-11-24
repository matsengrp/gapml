from typing import List, Dict, Set
import random
import logging
import numpy as np

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
            parent_targ_statuses: List[TargetStatus],
            anc_state: AncState,
            is_leaf: bool):
        self.target_tract_tuples = target_tract_tuples
        self.parent_targ_statuses = parent_targ_statuses
        # The possible states that precede the observed data at the leaves
        # Note that for leaf states, this is the set of all possible states that can precede this leaf,
        # even though the leaf can only be one state.
        target_statuses = list(set([
            TargetStatus.from_target_tract_tuple(tt_tuple)
            for tt_tuple in target_tract_tuples] + self.parent_targ_statuses))
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
            max_extra_steps: int = 1):
        """
        @param tree: the tree to create transition wrappers for
        @param max_extra_steps: number of extra steps to search for possible ancestral states
        """
        self.bcode_meta = bcode_metadata
        self.tree = tree

        self.max_extra_steps = max_extra_steps

    def create_transition_wrappers(self):
        """
        @return Dict[node id, List[TransitionWrapperMaker]]: a dictionary that stores the
                list of TransitionWrapperMakers (one for each barcode) for each node
        """
        # Annotate with ancestral states using the efficient upper bounding algo
        anc_evt_finder.annotate_ancestral_states(self.tree, self.bcode_meta)
        max_parsimony_sgs = anc_evt_finder.get_max_parsimony_anc_singletons(self.tree)

        # Create a dictionary mapping node to its TransitionWrapper
        transition_matrix_states = dict()
        for node in self.tree.traverse("preorder"):
            wrapper_list = []
            for idx, anc_state in enumerate(node.anc_state_list):
                if node.is_root():
                    wrapper_list.append(TransitionWrapper(
                        [TargetTractTuple()],
                        [],
                        anc_state,
                        node.is_leaf()))
                    continue

                # Subset the possible internal states as lower bound
                # The subset is going to be those within a small number of steps
                transition_matrix_states[node.up.node_id][idx]
                parent_target_tract_tuples = transition_matrix_states[node.up.node_id][idx].target_tract_tuples
                up_anc_state = node.up.anc_state_list[idx]
                minimal_sgs, maximal_sgs = max_parsimony_sgs[node.node_id][idx]
                minimal_up_sgs, maximal_up_sgs = max_parsimony_sgs[node.up.node_id][idx]
                min_required_steps = len(set(minimal_sgs) - set(maximal_up_sgs))
                max_required_steps = len(set(maximal_sgs) - set(minimal_up_sgs))
                requested_target_status = TargetStatus.from_target_tract_tuple(TargetTractTuple(*[sg.get_target_tract() for sg in minimal_sgs]))
                close_target_tract_tuples = self.get_states_close_by(
                        max_required_steps + self.max_extra_steps,
                        min_required_steps,
                        parent_target_tract_tuples,
                        anc_state,
                        requested_target_status)

                transition_wrap = TransitionWrapper(
                    close_target_tract_tuples,
                    [],
                    anc_state,
                    node.is_leaf())

                logging.info(
                    "Subsampling states for node %d, parent # states %d, node # subsampled states %d, (node target tract tuples %d)",
                    node.node_id,
                    len(transition_matrix_states[node.up.node_id][idx].states),
                    len(transition_wrap.states),
                    len(transition_wrap.target_tract_tuples))
                if len(transition_wrap.states) > 100:
                    logging.info("=== many-state-node %s", anc_state)
                    logging.info("=== parent-many-state-node %s", up_anc_state)

                wrapper_list.append(transition_wrap)

            transition_matrix_states[node.node_id] = wrapper_list
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

        TODO: this code is really inefficient and slow but it works for now

        @return List[TargetTractTuples] -- target tract tuples within max_steps of
        """
        if max_steps == 0:
            # special case: there are no steps we are allowed to take.
            # then the only possible state is exactly the state of the node
            # this should really only happen for nodes with no events or leaf nodes
            assert all([indel_set.__class__ == SingletonWC for indel_set in anc_state.indel_set_list])
            no_step_max_state = [sg.get_target_tract() for sg in anc_state.get_singleton_wcs()]
            return [TargetTractTuple(*no_step_max_state)]

        # map each target tract tuple to a Dict. Dict maps distance to paths of that distance to reach that target tract tuple
        dist_to_start_dict = {p: {0: [[p]]} for p in parent_target_tract_tuples}
        # Maps distance to nodes that we can reach in exactly that distance
        dist_to_states_dict = {0: [p for p in parent_target_tract_tuples]}
        max_targets = set(anc_state.to_max_target_status().get_inactive_targets(self.bcode_meta))
        # Keep track of the latest states visited
        states_to_explore = set(parent_target_tract_tuples)
        # Find all paths of at most max_steps long where paths are all possible according to `anc_state`
        for i in range(max_steps):
            dist_to_states_dict[i + 1] = []
            new_states = set()
            for state in states_to_explore:
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

                    # If this is first time we reached this state or this is one of the shortest paths to reach this state
                    if new_state not in dist_to_start_dict:
                        dist_to_start_dict[new_state] = {}
                    if i + 1 not in dist_to_start_dict[new_state]:
                        dist_to_start_dict[new_state][i + 1] = []
                    dist_to_start_dict[new_state][i + 1] += [path + [new_state] for path in dist_to_start_dict[state][i]]
                    dist_to_states_dict[i + 1].append(new_state)
                    new_states.add(new_state)
            states_to_explore = new_states

        # Pick out states along paths that reach the max singleton ancestral state
        # for this node within the specified `max_steps`
        sg_max_inactive_targs = requested_target_status.get_inactive_targets(self.bcode_meta)
        close_states = []
        for idx in range(min_steps_to_sg, max_steps + 1):
            for max_step_state in dist_to_states_dict[idx]:
                for path in dist_to_start_dict[max_step_state][idx]:
                    # Only need to start looking after `min_steps_to_sg` since we know this is the miin
                    # number of steps needed. Just for efficiency
                    for path_state in path[min_steps_to_sg:]:
                        path_targ_stat = TargetStatus.from_target_tract_tuple(path_state)
                        path_inactive_targs = path_targ_stat.get_inactive_targets(self.bcode_meta)
                        if set(sg_max_inactive_targs) <= set(path_inactive_targs):
                            close_states += path
                            break

        assert len(close_states) > 0
        return list(set(close_states))
