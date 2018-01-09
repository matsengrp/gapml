from typing import Tuple, List, Set, Dict
from functools import reduce
import itertools

from indel_sets import TargetTract, AncState, IndelSet, SingletonWC
from cell_lineage_tree import CellLineageTree
from state_sum import StateSum
from barcode_metadata import BarcodeMetadata

class TransitionToNode:
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
        return tt_group_new

class TransitionGraph:
    def __init__(self, edges: Dict[Tuple[TargetTract], Set[TransitionToNode]] = dict()):
        self.edges = edges

    def get_nodes(self):
        return self.edges.keys()

    def add_edge(self, from_node: Tuple[TargetTract], tt_evt: TargetTract):
        to_node_annotated = TransitionToNode.create_transition_to(from_node, tt_evt)
        self.edges[from_node].add(to_node_annotated)

    def add_node(self, node: Tuple[TargetTract]):
        if node in self.get_nodes():
            raise ValueError("Node already exists")
        else:
            self.edges[node] = set()

    def __str__(self):
        return str(self.edges)

class ApproximatorLB:
    def __init__(self, extra_steps: int, anc_generations: int):
        self.extra_steps = extra_steps
        self.anc_generations = anc_generations

    def annotate_state_sum_transitions(self, tree: CellLineageTree):
        for node in tree.traverse("preorder"):
            if node.is_root():
                node.add_feature("state_sum", StateSum())
            elif node.is_leaf():
                # TODO: Deal with a leaf which only has one possible state
                continue
            else:
                # TODO: make this look up self.anc_generations rather than only one
                anc = node.up
                print(anc.is_root())
                print(anc.state_sum)
                transition_graph, state_sum = self.get_branch_state_sum_transitions(anc, node)
                node.add_feature("state_sum", state_sum)
                node.add_feature("transition_graph", transition_graph)

    def get_branch_state_sum_transitions(self, anc: CellLineageTree, node: CellLineageTree):
        if len(node.anc_state.indel_set_list) == 0:
            # If unmodified
            return TransitionGraph({(): set()}), anc.state_sum

        tts_partitions = []
        for tts in anc.state_sum.tts_list:
            # Partition the TTs first according to node's anc_state
            tts_part = ApproximatorLB.partition(tts, node.anc_state)
            print(tts_part, "tts_part")
            tts_partitions.append(tts_part)

        subgraphs = []
        sub_state_sums = []
        anc_anc_state_partition = ApproximatorLB.partition(anc.anc_state.indel_set_list, node.anc_state)
        for max_indel_set in node.anc_state.indel_set_list:
            tt_tuple_starts = [tt_p[max_indel_set] for tt_p in tts_partitions]
            # Find each subgraph of TT groups that are nearby
            tt_tuple_subgraph = self.walk_tt_group_subgraph(max_indel_set, tt_tuple_starts)
            subgraphs.append(tt_tuple_subgraph)
            print(tt_tuple_subgraph)

            # Determine which tt_tuples will contribute to node's state sum
            # This means we want the tt_tuples that are not in anc's anc_state
            # To do this, it suffices to check which targets have been disabled
            anc_indel_set_tuple = anc_anc_state_partition[max_indel_set]
            anc_avail_targs = ApproximatorLB.get_avail_actions(max_indel_set, anc_indel_set_tuple)
            anc_avail_targs = set(anc_avail_targs)
            sub_state_sum = []
            for tt_tuple in tt_tuple_subgraph.get_nodes():
                print(tt_tuple, "tttt")
                deact_targs = ApproximatorLB.get_deactivated_targs(tt_tuple)
                # TODO: a faster version of this check please. don't need to get hte actual intersection
                if not deact_targs.intersection(anc_avail_targs):
                    sub_state_sum.append(tt_tuple)
            sub_state_sums.append(sub_state_sum)
            print(node.is_leaf(), "is leaf?")
            print("subbb", sub_state_sums)

        # Finally take the "product" of these sub_state_sums to form state sum
        product_state_sums = itertools.product(*sub_state_sums)
        node_state_sum = StateSum([
            reduce(lambda x,y: x + y, tup_tt_groups) for tup_tt_groups in product_state_sums
        ])

        return subgraphs, node_state_sum

    @staticmethod
    def get_deactivated_targs(tt_grp: Tuple[IndelSet]):
        if tt_grp:
            deactivated = range(tt_grp[0].min_deact_target, tt_grp[0].max_deact_target + 1)
            for tt in tt_grp[1:]:
                deactivated += range(tt.min_deact_target, tt.max_deact_target + 1)
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


    def walk_tt_group_subgraph(self, max_indel_set: IndelSet, tt_grp_roots: Set[Tuple[TargetTract]]):
        tt_group_graph = TransitionGraph()
        # Tracks which TT groups are within i steps
        tt_group_steps = [set() for i in range(self.extra_steps + 1)]
        # Tracks which TT groups have been visited already
        visited_tt_groups = set()

        def update_with_state(i, new_tt_grp: Tuple[TargetTract]):
            if new_tt_grp not in visited_tt_groups:
                visited_tt_groups.add(new_tt_grp)
                tt_group_steps[i].add(new_tt_grp)
                tt_group_graph.add_node(new_tt_grp)

        # For each i, find groups within i + 1 steps
        # TODO: make this more efficient
        for tt_grp_rt in tt_grp_roots:
            update_with_state(0, tt_grp_rt)

        for i in range(self.extra_steps):
            if i == 0:
                # Special case dealing with singleton
                singleton = max_indel_set.get_singleton()
                if singleton is not None:
                    singleton_tt = TargetTract(
                        singleton.min_deact_target,
                        singleton.min_target,
                        singleton.max_target,
                        singleton.max_deact_target)
                    update_with_state(i + 1, (singleton_tt,))

            # Now consider all the elements that are i steps away from the root
            # Find things that are one more step away
            for tt_grp_start in tt_group_steps[i]:
                # Find available actions
                active_any_targs = ApproximatorLB.get_avail_actions(max_indel_set, tt_grp_start)
                n_any_targs = len(active_any_targs)

                # Take one step from this TT group
                # Create possible starts of the target tracts
                all_starts = [[] for _ in range(n_any_targs)]
                for i0_prime, t0_prime in enumerate(active_any_targs):
                    # No left trim overflow
                    all_starts[i0_prime].append((t0_prime, t0_prime))
                    if active_any_targs[i0_prime + 1] == t0_prime + 1:
                        # Add in the long left trim overflow
                        all_starts[i0_prime + 1].append((t0_prime, t0_prime + 1))

                # Create possible ends of the target tracts
                all_ends = [[] for i in range(n_any_targs)]
                for i1_prime, t1_prime in enumerate(active_any_targs):
                    # No right trim overflow
                    all_ends[i1_prime].append((t1_prime, t1_prime))
                    if active_any_targs[i1_prime - 1] == t1_prime - 1:
                        # Add in the right trim overflow
                        all_ends[i1_prime - 1].append((t1_prime - 1, t1_prime))

                # Finally create all possible target tracts by combining possible start and ends
                for j, tt_starts in enumerate(all_starts):
                    for k in range(j, n_any_targs):
                        tt_ends = all_ends[k]
                        for tt_start in tt_starts:
                            for tt_end in tt_ends:
                                one_step_tt = TargetTract(tt_start[0], tt_start[1], tt_end[0], tt_end[1])
                                update_with_state(i + 1, one_step_tt)
                                tt_group_graph.add_edge(
                                    tt_grp_start,
                                    one_step_tt)

                # Finally add in the singleton as a possible move
                if singleton is not None and singleton_tt != tt_grp_start:
                    tt_group_graph.add_edge(
                        tt_grp_start,
                        singleton_tt)

        return tt_group_graph

    @staticmethod
    def get_avail_actions(indel_set: IndelSet, tt_grp: Tuple[TargetTract]):
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
                active_any_targs = range(wc.min_target, tt.min_deact_target)
                for i, tt in enumerate(tt_grp[1:]):
                    active_targets += range(tt_grp[i - 1].max_deact.target + 1, tt.min_deact.target)
                active_any_targs += range(tt.max_deact.target + 1, wc.max_target + 1)
            else:
                active_any_targs = range(wc.min_target, wc.max_target)

        return active_any_targs

