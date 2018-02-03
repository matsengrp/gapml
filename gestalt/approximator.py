from typing import Tuple, List, Set, Dict
import itertools

from indel_sets import TargetTract, AncState, IndelSet, SingletonWC
from cell_lineage_tree import CellLineageTree
from state_sum import StateSum
from barcode_metadata import BarcodeMetadata
from common import merge_target_tract_groups
from clt_likelihood_model import CLTLikelihoodModel
from constants import UNLIKELY
from transition_matrix import TransitionMatrixWrapper

from approximator_transition_graph import TransitionToNode, TransitionGraph

class ApproximatorLB:
    """
    Class that helps approximate the likelihood via a lower bound.
    The lower bound is created by summing over a subset of the possible ancestral states at each node.
    The states we sum over for each node is stored in StateSum.
    The approximation algo is parameterized by:
    1. extra_steps: the number of steps from the StateSum of the ancestor node
    2. anc_generations: the number of generations above the current node to use as a "lower bound" of the StateSum of the current node
    """
    def __init__(self, extra_steps: int, anc_generations: int, bcode_metadata: BarcodeMetadata):
        self.extra_steps = extra_steps
        self.anc_generations = anc_generations
        self.bcode_meta = bcode_metadata

    def create_transition_matrix_wrappers(self, topology: CellLineageTree):
        """
        Create a skeleton of the transition matrix for each branch using the approximation algo

        @param topology: a tree, assumes this tree has the "node_id" feature
        @return dictionary mapping `node_id` to TransitionMatrixWrapper
        """
        # First determine the state sum and transition possibilities of each node
        self._annotate_state_sum_transitions(topology)

        # Now create the TransitionMatrixWrapper for each node
        transition_matrix_wrappers = dict()
        for node in topology.traverse("postorder"):
            if not node.is_root():
                trans_mat = self._create_transition_matrix_wrapper(node)
                transition_matrix_wrappers[node.node_id] = trans_mat
        return transition_matrix_wrappers

    def _annotate_state_sum_transitions(self, tree: CellLineageTree):
        """
        Annotate each branch of the tree with the state sum.
        Also annotates with the transition graphs between target tract representations.

        TODO: do not annotate the tree, instead just return a dictionary?

        The `state_sum` attribute is a list of target tract reprs that are in StateSum for that node.
        The `transition_graph_dict` attribute is a dictionary:
          key = a tuple of target tracts (corresponds to a subset of a target tract representation)
                AND the maximum target tract that tuple can transition to
          value = the transition graphs starting at the tuple
        We use transition_graph_dict later to construct all the possible states in the
        transition matrix.
        """
        for node in tree.traverse("preorder"):
            if node.is_root():
                node.add_feature("state_sum", StateSum([()]))
            else:
                transition_graph_dict, state_sum = self._get_branch_state_sum_transitions(node)
                if node.is_leaf():
                    state_sum = StateSum.create_for_observed_allele(node.allele_events, self.bcode_meta)
                node.add_feature("state_sum", state_sum)
                node.add_feature("transition_graph_dict", transition_graph_dict)

    def _get_branch_state_sum_transitions(self, node: CellLineageTree):
        """
        @param node: the node we are getting the state sum for
        @return a dictionary of transition graphs between groups of target tracts,
                indexed by their start group of target tracts and the maximum target tract
                AND the state sum for this node
        """
        anc = node.up_generations(self.anc_generations)
        if len(node.anc_state.indel_set_list) == 0:
            # If unmodified
            return dict(), anc.state_sum

        # Partition the anc's anc_state according to node's anc_state.
        # Used for deciding which states to add in node's StateSum since node's StateSum is
        # composed of states that cannot be < ancestor node's anc_state.
        anc_partition = self.partition(anc.anc_state.indel_set_list, node.anc_state)

        # For each state in the parent node's StateSum, find the subgraph of nearby target tract repr
        node_state_sum = set()
        # Stores previously-explored subgraphs from a particular root node and max_indel_set
        subgraph_dict = {}
        # Stores previously-calculated partitioned state sums and max_indel_set
        sub_state_sums_dict = {}
        for tts in node.up.state_sum.tts_list:
            # Partition each state in the ancestral node's StateSum according to node's anc_state
            tts_partition = self.partition(tts, node.anc_state)
            tts_sub_state_sums = []
            for max_indel_set in node.anc_state.indel_set_list:
                tt_tuple_start = tts_partition[max_indel_set]
                graph_key = (tt_tuple_start, max_indel_set)
                if graph_key not in subgraph_dict:
                    # We have never tried walking the subgraph for this start node, so let's do it now.
                    subgraph = self._walk_tt_group_subgraph(max_indel_set, tt_tuple_start)
                    subgraph_dict[graph_key] = subgraph

                    # Determine which tt_tuples will contribute to node's state sum
                    # This means we want the tt_tuples that are not in anc's anc_state
                    # To do this, it suffices to check which targets have been disabled
                    anc_indel_set_tuple = anc_partition[max_indel_set]
                    anc_deact_targs = ApproximatorLB.get_deactivated_targets(anc_indel_set_tuple)
                    sub_state_sum = set()
                    for tt_tuple in subgraph.get_nodes():
                        deact_targs = ApproximatorLB.get_deactivated_targets(tt_tuple)
                        if deact_targs >= anc_deact_targs:
                            sub_state_sum.add(tt_tuple)
                    sub_state_sums_dict[graph_key] = sub_state_sum

                tts_sub_state_sums.append(sub_state_sums_dict[graph_key])

            # Finally take the "product" of these sub_state_sums to form state sum
            product_state_sums = itertools.product(*tts_sub_state_sums)
            node_state_sum.update([
                merge_target_tract_groups(tup_tt_groups) for tup_tt_groups in product_state_sums
            ])

        return subgraph_dict, StateSum(node_state_sum)

    def _walk_tt_group_subgraph(self, max_indel_set: IndelSet, tt_grp_rt: Tuple[TargetTract]):
        """
        Dynamic programming algo for finding subgraph of nodes that are within `self.extra_steps` of
        any node in `tt_grp_roots`.

        @param max_indel_set: specifies the maximum you can cut
        @param tt_grp_roots: create the subgraph starting at these nodes
        """
        tt_group_graph = TransitionGraph(dict())

        # Tracks which TT groups are within i steps
        tt_group_steps = [set() for i in range(self.extra_steps + 1)]
        # Tracks which TT groups have been visited already
        visited_tt_groups = set()

        def update_with_state(i: int, new_tt_grp: Tuple[TargetTract]):
            if new_tt_grp not in visited_tt_groups:
                visited_tt_groups.add(new_tt_grp)
                tt_group_steps[i].add(new_tt_grp)
                tt_group_graph.add_node(new_tt_grp)

        # Add the zero-step node
        update_with_state(0, tt_grp_rt)

        # Create the singleton TT -- this singleton can be reached in one step
        singleton = max_indel_set.get_singleton()
        if singleton is not None:
            singleton_tt = TargetTract(
                singleton.min_deact_target,
                singleton.min_target,
                singleton.max_target,
                singleton.max_deact_target)

        # Find groups within i > 0 steps
        for i in range(1, 1 + self.extra_steps):
            # Now consider all the elements that are i - 1 steps away from the root
            # Find things that are one more step away
            for tt_grp_start in tt_group_steps[i - 1]:
                # Find available actions
                active_any_targs = ApproximatorLB.get_active_any_trim_targets(max_indel_set, tt_grp_start)
                # Get possible target tract events
                tt_evts = CLTLikelihoodModel.get_possible_target_tracts(active_any_targs)
                # Add possible target tract events to the graph
                for tt_evt in tt_evts:
                    to_node = TransitionToNode.create_transition_to(tt_grp_start, tt_evt)
                    update_with_state(i, to_node.tt_group)
                    tt_group_graph.add_edge(tt_grp_start, to_node)

                # Finally add in the singleton as a possible move
                if singleton is not None and (singleton_tt,) != tt_grp_start:
                    to_node = TransitionToNode(singleton_tt, (singleton_tt, ))
                    update_with_state(i, to_node.tt_group)
                    tt_group_graph.add_edge(tt_grp_start, to_node)

        return tt_group_graph

    def _create_transition_matrix_wrapper(self, node: CellLineageTree):
        """
        @return TransitionMatrixWrapper for the particular branch ending at `node`
                only contains the states relevant to state_sum
        """
        transition_dict = dict()
        indel_set_list = node.anc_state.indel_set_list
        # Determine the values in the transition matrix by considering all possible states
        # starting at the parent's StateSum.
        # Recurse through all of its children to build out the transition matrix
        for tts in node.up.state_sum.tts_list:
            tts_partition_info = dict()
            tts_partition = self.partition(tts, node.anc_state)
            for indel_set in indel_set_list:
                tt_tuple = tts_partition[indel_set]
                graph_key = (tt_tuple, indel_set)
                # To recurse, indicate the subgraphs for each partition and the current node
                # (target tract group) we are currently located at.
                tts_partition_info[indel_set] = {
                        "start": tt_tuple,
                        "graph": node.transition_graph_dict[graph_key]}
            self._add_transition_dict_row(tts_partition_info, indel_set_list, transition_dict)

        # Create sparse transition matrix given the dictionary representation
        return TransitionMatrixWrapper(transition_dict)

    def _add_transition_dict_row(
            self,
            tts_partition_info: Dict[IndelSet, Dict],
            indel_set_list: List[IndelSet],
            transition_dict: Dict):
        """
        Recursive function for adding transition matrix rows.
        Function will modify transition_dict.
        The rows added will correspond to states that are reachable along the subgraphs
        in tts_partition_info.

        @param tts_partition_info: indicate the subgraphs for each partition and the current node
                                we are at for each subgraph
        @param indel_set_list: the ordered list of indel sets from the node's AncState
        @param transtion_dict: the dictionary corresponding to transitions
        """
        start_tts = merge_target_tract_groups([
            tts_partition_info[ind_set]["start"] for ind_set in indel_set_list])
        transition_dict[start_tts] = dict()

        # Find all possible target tract representations within one step of start_tts
        # Do this by taking one step in one of the subgraphs
        for indel_set, val in tts_partition_info.items():
            subgraph = val["graph"]
            tt_tuple_start = val["start"]

            # Each child is a possibility
            children = subgraph.get_children(tt_tuple_start)
            for child in children:
                new_tts_part_info = {k: v.copy() for k,v in tts_partition_info.items()}
                new_tts_part_info[indel_set]["start"] = child.tt_group

                # Create the new target tract representation
                new_tts = merge_target_tract_groups([
                    new_tts_part_info[ind_set]["start"] for ind_set in indel_set_list])

                # Add entry to transition matrix
                if new_tts not in transition_dict[start_tts]:
                    transition_dict[start_tts][new_tts] = child.tt_evt
                else:
                    raise ValueError("already exists?")

                # Recurse
                self._add_transition_dict_row(new_tts_part_info, indel_set_list, transition_dict)

    @staticmethod
    def get_deactivated_targets(tt_grp: Tuple[IndelSet]):
        if tt_grp:
            deactivated = list(range(tt_grp[0].min_deact_target, tt_grp[0].max_deact_target + 1))
            for tt in tt_grp[1:]:
                deactivated += list(range(tt.min_deact_target, tt.max_deact_target + 1))
            return set(deactivated)
        else:
            return set()

    @staticmethod
    def get_active_any_trim_targets(indel_set: IndelSet, tt_grp: Tuple[TargetTract]):
        """
        @return
            list of active targets that can be cut in any manner
        """
        wc = indel_set.inner_wc

        # Stores which targets can be cut in any manner
        active_any_targs = []
        if wc:
            if tt_grp:
                tt = tt_grp[0]
                active_any_targs = list(range(wc.min_target, tt.min_deact_target))
                for i, tt in enumerate(tt_grp[1:]):
                    active_any_targs += list(range(tt_grp[i].max_deact_target + 1, tt.min_deact_target))
                active_any_targs += list(range(tt.max_deact_target + 1, wc.max_target + 1))
            else:
                active_any_targs = list(range(wc.min_target, wc.max_target + 1))

        return active_any_targs

    @staticmethod
    def partition(tts: Tuple[IndelSet], anc_state: AncState):
        """
        @return split tts according to anc_state: Dict[IndelSet, Tuple[TargetTract]]
        """
        parts = {indel_set : () for indel_set in anc_state.indel_set_list}

        tts_idx = 0
        n_tt = len(tts)
        anc_state_idx = 0
        n_anc_state = len(anc_state.indel_set_list)
        while tts_idx < n_tt and anc_state_idx < n_anc_state:
            cur_tt = tts[tts_idx]
            indel_set = anc_state.indel_set_list[anc_state_idx]

            if cur_tt.max_deact_target < indel_set.min_deact_target:
                tts_idx += 1
                continue
            elif indel_set.max_deact_target < cur_tt.min_deact_target:
                anc_state_idx += 1
                continue

            # Should be overlapping now
            parts[indel_set] = parts[indel_set] + (cur_tt,)
            tts_idx += 1
        return parts
