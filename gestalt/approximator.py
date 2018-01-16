from typing import Tuple, List, Set, Dict
from functools import reduce
import itertools

from indel_sets import TargetTract, AncState, IndelSet, SingletonWC
from cell_lineage_tree import CellLineageTree
from state_sum import StateSum
from barcode_metadata import BarcodeMetadata

class TransitionToNode:
    """
    Stores information on the transition to another node in the TransitionGraph
    """
    def __init__(self, tt_evt: TargetTract, tt_group: Tuple[TargetTract]):
        """
        @param tt_evt: the target tract that was introduced
        @param tt_group: the resulting target tract after this was introduced
        """
        self.tt_evt = tt_evt
        self.tt_group = tt_group

    @staticmethod
    def create_transition_to(tt_group_orig: Tuple[TargetTract], tt_evt: TargetTract):
        """
        @param tt_group_orig: the original target tract group
        @param tt_evt: the added target tract
        @return the new Tuple[TargetTract]
        """
        # TODO: check that this target tract can be added?
        if len(tt_group_orig):
            tt_group_new = ()
            tt_evt_added = False
            for i, tt in enumerate(tt_group_orig):
                if tt.max_deact_target < tt_evt.min_deact_target or tt_evt.max_deact_target < tt.min_deact_target:
                    tt_group_new += (tt,)

                if i + 1 < len(tt_group_orig):
                    next_tt = tt_group_orig[i + 1]
                    if next_tt.min_deact_target > tt_evt.max_deact_target:
                        tt_group_new += (tt_evt,)
                        tt_evt_added = True
            return TransitionToNode(tt_evt, tt_group_new)
        else:
            return TransitionToNode(tt_evt, (tt_evt,))

class TransitionGraph:
    """
    The graph of how (partitions of) target tract representations can transition to each other.
    Stores edges as a dictionary where the key is the from node and the values are the directed edges from that node.
    """
    def __init__(self, edges: Dict[Tuple[TargetTract], Set[TransitionToNode]] = dict()):
        self.edges = edges

    def get_nodes(self):
        return self.edges.keys()

    def add_edge(self, from_node: Tuple[TargetTract], to_node: TransitionToNode):
        if to_node in self.edges[from_node]:
            raise ValueError("Edge already exists?")
        self.edges[from_node].add(to_node)

    def add_node(self, node: Tuple[TargetTract]):
        if len(self.edges) and node in self.get_nodes():
            raise ValueError("Node already exists")
        else:
            self.edges[node] = set()

    def __str__(self):
        return str(self.edges)

class ApproximatorLB:
    """
    Class that helps approximate the likelihood via a lower bound.
    The lower bound is created by summing over a subset of the possible ancestral states at each node.
    The states we sum over for each node is stored in StateSum.
    The approximation algo is parameterized by:
    1. extra_steps: the number of steps from the StateSum of the ancestor node
    2. anc_generations: the number of generations above the current node to use as a "lower bound" of the StateSum of the current node
    """
    def __init__(self, extra_steps: int, anc_generations: int):
        self.extra_steps = extra_steps
        self.anc_generations = anc_generations

    def annotate_state_sum_transitions(self, tree: CellLineageTree):
        for node in tree.traverse("preorder"):
            if node.is_root():
                node.add_feature("state_sum", StateSum(set([()])))
            elif node.is_leaf():
                # TODO: Deal with a leaf which only has one possible state
                continue
            else:
                # TODO: make this look up self.anc_generations rather than only one
                anc = node.up
                transition_graph_dict, state_sum = self.get_branch_state_sum_transitions(anc, node)
                node.add_feature("state_sum", state_sum)
                node.add_feature("transition_graph_dict", transition_graph_dict)

    def get_branch_state_sum_transitions(self, anc: CellLineageTree, node: CellLineageTree):
        if len(node.anc_state.indel_set_list) == 0:
            # If unmodified
            return dict(), anc.state_sum

        # Partition the ancestral node's anc_state according to node's anc_state.
        # Used for deciding which states to add in node's StateSum since node's StateSum is
        # composed of states that cannot be < ancestor node's anc_state.
        anc_partition = ApproximatorLB.partition(anc.anc_state.indel_set_list, node.anc_state)

        # For each state in in the ancestral node's StateSum, find the subgraph of nearby target tract repr
        node_state_sum = StateSum(set())
        # Stores previously-explored subgraphs from a particular root node and max_indel_set
        subgraph_dict = {}
        # Stores previously-calculated partitioned state sums and max_indel_set
        sub_state_sums_dict = {}
        for tts in anc.state_sum.tts_set:
            # Partition each state in the ancestral node's StateSum according to node's anc_state
            tts_partition = ApproximatorLB.partition(tts, node.anc_state)
            tts_sub_state_sums = []
            for max_indel_set in node.anc_state.indel_set_list:
                tt_tuple_start = tts_partition[max_indel_set]
                graph_key = (tt_tuple_start, max_indel_set)
                if graph_key not in subgraph_dict:
                    # We have never tried walking the subgraph for this start node, so let's do it now.
                    subgraph = self.walk_tt_group_subgraph(max_indel_set, tt_tuple_start)
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
                reduce(lambda x,y: x + y, tup_tt_groups) for tup_tt_groups in product_state_sums
            ])

        return subgraph_dict, node_state_sum

    def walk_tt_group_subgraph(self, max_indel_set: IndelSet, tt_grp_rt: Tuple[TargetTract]):
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
                tt_evts = ApproximatorLB.get_possible_any_trim_target_tracts(active_any_targs)
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
    def partition(tts: Tuple[IndelSet], anc_state: AncState):
        """
        TODO: maybe move this to some other function?
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
    def get_possible_any_trim_target_tracts(active_any_targs: List[int]):
        """
        @param active_any_targs: a list of active targets that can be cut with any trim
        @return a set of possible target tracts
        """
        n_any_targs = len(active_any_targs)

        # Take one step from this TT group using two step procedure
        # 1. enumerate all possible start positions for target tract
        # 2. enumerate all possible end positions for target tract

        # List possible starts of the target tracts
        all_starts = [[] for _ in range(n_any_targs)]
        for i0_prime, t0_prime in enumerate(active_any_targs):
            # No left trim overflow
            all_starts[i0_prime].append((t0_prime, t0_prime))
            # Determine if left trim overflow allowed
            if i0_prime < n_any_targs - 1 and active_any_targs[i0_prime + 1] == t0_prime + 1:
                # Add in the long left trim overflow
                all_starts[i0_prime + 1].append((t0_prime, t0_prime + 1))

        # Create possible ends of the target tracts
        all_ends = [[] for i in range(n_any_targs)]
        for i1_prime, t1_prime in enumerate(active_any_targs):
            # No right trim overflow
            all_ends[i1_prime].append((t1_prime, t1_prime))
            # Determine if right trim overflow allowed
            if i1_prime > 0 and active_any_targs[i1_prime - 1] == t1_prime - 1:
                # Add in the right trim overflow
                all_ends[i1_prime - 1].append((t1_prime - 1, t1_prime))

        # Finally create all possible target tracts by combining possible start and ends
        tt_evts = set()
        for j, tt_starts in enumerate(all_starts):
            for k in range(j, n_any_targs):
                tt_ends = all_ends[k]
                for tt_start in tt_starts:
                    for tt_end in tt_ends:
                        tt_evt = TargetTract(tt_start[0], tt_start[1], tt_end[0], tt_end[1])
                        tt_evts.add(tt_evt)

        return tt_evts
