from typing import Tuple, List, Set, Dict

from indel_sets import TargetTract, Tract, DeactTract, DeactTargetsEvt

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

    def __str__(self):
        return "%s, %s" % (self.tt_evt, self.tt_group)

    @staticmethod
    def create_transition_to(tt_group_orig: Tuple[Tract], tt_evt: Tract):
        """
        @param tt_group_orig: the original target tract group
        @param tt_evt: the added target tract
        @return the new Tuple[Tract]
        """
        # TODO: check that this target tract can be added?
        tt_result = tt_evt if tt_evt.__class__ == TargetTract else DeactTract(
                tt_evt.min_deact_target, tt_evt.max_deact_target)

        if len(tt_group_orig):
            tt_group_new = ()
            tt_evt_added = False
            for i, tt in enumerate(tt_group_orig):
                if i == 0:
                    if tt_evt.max_deact_target < tt.min_deact_target:
                        tt_group_new += (tt_result,)
                        tt_evt_added = True

                if tt.max_deact_target < tt_evt.min_deact_target or tt_evt.max_deact_target < tt.min_deact_target:
                    tt_group_new += (tt,)

                if not tt_evt_added and i < len(tt_group_orig) - 1:
                    next_tt = tt_group_orig[i + 1]
                    if next_tt.min_deact_target > tt_evt.max_deact_target:
                        tt_group_new += (tt_result,)
                        tt_evt_added = True
            if not tt_evt_added:
                tt_group_new += (tt_result,)

            # TODO: factorize the code and make a test for this
            # Now merge adjacent deact tracts
            tract_group_merged = ()
            curr_deact_tract = None
            for tract in tt_group_new:
                if tract.is_target_tract:
                    if curr_deact_tract:
                        tract_group_merged += (curr_deact_tract, )
                        curr_deact_tract = None

                    tract_group_merged += (tract,)
                else:
                    if curr_deact_tract is None:
                        curr_deact_tract = tract
                    elif tract.min_deact_target == curr_deact_tract.max_deact_target + 1:
                        curr_deact_tract = DeactTract(
                                curr_deact_tract.min_deact_target,
                                tract.max_deact_target)
                    else:
                        tract_group_merged += (curr_deact_tract, )
                        curr_deact_tract = tract

            # If there is a lingering deact tract to add to the list...
            if curr_deact_tract:
                tract_group_merged += (curr_deact_tract, )

            print("tract_group_merged", tract_group_merged)

            return TransitionToNode(tt_evt, tract_group_merged)
        else:
            return TransitionToNode(tt_evt, (tt_result,))

class TransitionGraph:
    """
    The graph of how (partitions of) target tract representations can transition to each other.
    Stores edges as a dictionary where the key is the from node and the values are the directed edges from that node.
    """
    def __init__(self, edges: Dict[Tuple[TargetTract], Set[TransitionToNode]] = dict()):
        self.edges = edges

    def get_children(self, node: Tuple[TargetTract]):
        return self.edges[node]

    def get_nodes(self):
        return self.edges.keys()

    def add_edge(self, from_node: Tuple[TargetTract], to_node: TransitionToNode):
        if to_node in self.edges[from_node]:
            raise ValueError("Edge already exists?")
        if to_node.tt_group == from_node:
            raise ValueError("Huh they are the same")
        self.edges[from_node].add(to_node)

    def add_node(self, node: Tuple[TargetTract]):
        if len(self.edges) and node in self.get_nodes():
            raise ValueError("Node already exists")
        else:
            self.edges[node] = set()

    def __str__(self):
        return str(self.edges)
