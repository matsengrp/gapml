import numpy as np
from typing import List, Dict

from allele_events import Event
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from split_data import create_kfold_trees, create_kfold_barcode_trees, TreeDataSplit

class HangingChad:
    def __init__(self,
            node: CellLineageTree,
            possible_parents: List[CellLineageTree],
            parsimony_contribution: int):
        self.node = node
        self.parsimony_contribution = parsimony_contribution
        self.possible_parents = possible_parents

def get_chads(tree: CellLineageTree):
    """
    @return Dict[Event, HangingChad]
    """
    hanging_chad_dict = dict()
    for node in tree.traverse("preorder"):
        if node.is_root():
            continue

        node_events = node.allele_events_list[0].events
        for evt in node_events:
            if evt.min_target + 2 <= evt.max_target:
                if evt not in hanging_chad_dict:
                    hanging_chad = get_possible_chad_parents(
                        tree, evt, node)
                    hanging_chad_dict[evt] = hanging_chad
    return hanging_chad_dict

def get_possible_chad_parents(
        tree: CellLineageTree,
        hanging_chad: Event,
        hanging_chad_node: CellLineageTree):
    """
    @return HangingChad
    """
    parent_events = set(hanging_chad_node.up.allele_events_list[0].events)
    chad_events = set(hanging_chad_node.allele_events_list[0].events)
    parsimony_contribution = len(chad_events - parent_events)

    possible_chad_locations = {}
    for node in tree.traverse("preorder"):
        # TODO: i dont think this finds all possible hanging chad locations
        # but it's good enough for our first stab
        node_events = set(node.allele_events_list[0].events)
        if node_events == chad_events:
            continue
        new_events = chad_events - node_events
        existing_events = node_events - chad_events
        hides_all_remain = all([
            hanging_chad.hides(remain_evt) for remain_evt in existing_events])
        potential_contribution = len(new_events)
        if hides_all_remain and potential_contribution <= parsimony_contribution:
            assert potential_contribution == parsimony_contribution
            if node.allele_events_list_str not in possible_chad_locations:
                possible_chad_locations[node.allele_events_list_str] = node
    #print("chad can go here", len(possible_chad_locations))
    #for evt_str, node in possible_chad_locations.items():
    #    print(evt_str, node.allele_events_list[0].events)
    #print(tree.get_ascii(attributes=["allele_events_list_str"]))
    return HangingChad(
            hanging_chad_node,
            list(possible_chad_locations.values()),
            parsimony_contribution)
