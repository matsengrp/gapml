from cell_lineage_tree import CellLineageTree

"""
This file will list functions that we can use for picking which target statuses to penalize.
"""

def mark_target_status_to_penalize(tree: CellLineageTree):
    """
    Takes the possible set of target statuses for the node's ancestral state and use
    its hazard rate of leaving those target statuses for penalization.
    """
    for node in tree.get_descendants():
        node.add_feature(
            'pen_targ_stat',
            [anc_state.generate_possible_target_statuses() for anc_state in node.anc_state_list])
