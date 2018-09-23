from cell_lineage_tree import CellLineageTree

"""
This file will list functions that we can use for picking which target statuses to penalize.
"""

def mark_target_status_to_penalize(tree: CellLineageTree):
    """
    Takes the maximal set of singletons from the ancestral state and uses
    that for penalization.
    As a result, we penalize the probability of leaving this state.
    For spines, we penalize the probability of leaving the spine's anc_state since
    MLE will want to assign the spine zero length for branches with no events.
    For all other nodes, we penalize the probability of leaving the parent's anc_state
    since the MLE will want to assign long lengths to branches with events.
    """
    for node in tree.get_descendants():
        which_node = node.up if node.is_resolved_multifurcation() else node
        node.add_feature(
            'pen_targ_stat',
            [anc_state.to_sg_max_target_status() for anc_state in which_node.anc_state_list])
