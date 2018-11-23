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
        # Annotate with ancestral states using the efficient upper bounding algo
        anc_evt_finder.annotate_ancestral_states(self.tree, self.bcode_meta)

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
                parent_target_tract_tuples = transition_matrix_states[node.up.node_id][idx].target_tract_tuples
                up_anc_state = node.up.anc_state_list[idx]
                min_required_steps = len(set(anc_state.get_singletons()) - set(up_anc_state.get_singletons()))
                close_target_tract_tuples = self.get_states_close_by(
                        min_required_steps + self.max_extra_steps,
                        min_required_steps,
                        parent_target_tract_tuples,
                        anc_state)

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
            anc_state: AncState):
        """
        @param parent_statuses: the parent target statuses
        @param targ_stat_transitions_dict: dictionary specifying all possible transitions between target statuses
        @param targ_stat_inv_transitions_dict: dictionary specifying all possible inverse/backwards transitions between target statuses
        @param end_target_status: the ending target status
        @param anc_state: AncState for specifying the possible ancestral states of this node

        @return Set[TargetTractTuples] -- target statuses that are within an EXTRA `self.max_extra_steps` steps away from `end_target_status`
                when starting at any state in `parent_statuses`. (extra meaning more than the minimum required to go from some parent
                state to the end state )
        """
        # label all nodes starting from the parent statuses with their closest distance (up to the max distance)
        dist_to_start_dict = {p: {"dist": 0, "paths": [[p]]} for p in parent_target_tract_tuples}
        states_to_explore = set(parent_target_tract_tuples)
        max_targets = set(anc_state.to_max_target_status().get_inactive_targets(self.bcode_meta))
        for i in range(max_steps):
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
                    if (new_state not in dist_to_start_dict) or (dist_to_start_dict[new_state]["dist"] >= i + 1):
                        new_paths = [path + [new_state] for path in dist_to_start_dict[state]["paths"]]
                        if (new_state not in dist_to_start_dict) or (dist_to_start_dict[new_state]["dist"] > i + 1):
                            combined_new_paths = new_paths
                        else:
                            combined_new_paths = new_paths + dist_to_start_dict[new_state]["paths"]
                        dist_to_start_dict[new_state] = {
                                "dist": i + 1,
                                "paths": combined_new_paths}
                        new_states.add(new_state)
            states_to_explore = new_states

        # Pick out states along paths that reach the max singleton ancestral state
        # for this node within the specified `max_steps`
        sg_max_targ_stat = anc_state.to_sg_max_target_status()
        sg_max_inactive_targs = sg_max_targ_stat.get_inactive_targets(self.bcode_meta)
        close_states = []
        for max_step_state in states_to_explore:
            for path in dist_to_start_dict[max_step_state]["paths"]:
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
