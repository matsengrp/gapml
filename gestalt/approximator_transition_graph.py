from typing import Tuple, List, Set, Dict

from indel_sets import TargetTract, Tract, DeactEvt, DeactTract, DeactTargetsEvt
from indel_sets import merge_tracts

class TransitionToNode:
    """
    Stores information on the transition to another node in the TransitionGraph
    """
    def __init__(self, deact_evt: DeactEvt, tract_group: Tuple[Tract]):
        """
        @param deact_evt: the target tract that was introduced
        @param tract_group: the resulting target tract after this was introduced
        """
        self.deact_evt = deact_evt
        self.tract_group = tract_group

    def __str__(self):
        return "%s, %s" % (self.deact_evt, self.tract_group)

    @staticmethod
    def create_transition_to(tract_group_orig: Tuple[Tract], deact_evt: DeactEvt):
        """
        @param tract_group_orig: the original tract group
        @param deact_evt: the new event
        @return the new Tuple[Tract]
        """
        tract_result = deact_evt.get_deact_result()

        # Now merge adjacent deact tracts
        tract_group_merged = merge_tracts(tract_group_orig, tract_result)
        return TransitionToNode(deact_evt, tract_group_merged)

class TransitionGraph:
    """
    The graph of how (partitions of) target tract representations can transition to each other.
    Stores edges as a dictionary where the key is the from node and the values are the directed edges from that node.
    """
    def __init__(self, edges: Dict[Tuple[DeactEvt], Set[TransitionToNode]] = dict()):
        self.edges = edges

    def get_children(self, node: Tuple[Tract]):
        return self.edges[node]

    def get_nodes(self):
        return self.edges.keys()

    def add_edge(self, from_node: Tuple[Tract], to_node: TransitionToNode):
        if to_node in self.edges[from_node]:
            raise ValueError("Edge already exists?")
        if to_node.tract_group == from_node:
            raise ValueError("Huh they are the same")
        self.edges[from_node].add(to_node)

    def add_node(self, node: Tuple[Tract]):
        if len(self.edges) and node in self.get_nodes():
            raise ValueError("Node already exists")
        else:
            self.edges[node] = set()

    def __str__(self):
        return str(self.edges)
