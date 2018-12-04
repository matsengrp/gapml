import logging
from functools import reduce

from cell_lineage_tree import CellLineageTree
from anc_state import AncState

from barcode_metadata import BarcodeMetadata

"""
Our in-built engine for finding all possible
events in the internal nodes.
"""

def annotate_ancestral_states(tree: CellLineageTree, bcode_meta: BarcodeMetadata, do_fast: bool=False):
    """
    Find all possible events in the internal nodes.
    @param do_fast: indicates whether to keep valid ancestral states alone and
                    only recalculate ancestral states for those that are set to None
    """
    # TODO: add python unit test for this fast version
    for node in tree.traverse("postorder"):
        if do_fast and node.anc_state_list is not None:
            continue

        if node.is_leaf():
            node.add_feature("anc_state_list", [
                AncState.create_for_observed_allele(evts, bcode_meta)
                for evts in node.allele_events_list])
        elif node.is_root():
            node.add_feature("anc_state_list", [AncState() for _ in range(bcode_meta.num_barcodes)])
        else:
            node_anc_state = get_possible_anc_states(node)
            node.add_feature("anc_state_list", node_anc_state)
        node.add_feature(
                "anc_state_list_str",
                "%s:%s" % (str(node.node_id), [str(k) for k in node.anc_state_list]))
    #logging.info("Ancestral state")
    #logging.info(node.get_ascii(attributes=["anc_state_list_str"], show_internal=True))


def get_possible_anc_states(tree: CellLineageTree):
    children = tree.get_children()
    parent_anc_state_list = []
    for bcode_idx, par_anc_state in enumerate(children[0].anc_state_list):
        for c in children[1:]:
            par_anc_state = AncState.intersect(
                par_anc_state,
                c.anc_state_list[bcode_idx])
        parent_anc_state_list.append(par_anc_state)
    return parent_anc_state_list


def get_parsimony_score(tree: CellLineageTree, do_fast: bool=False):
    """
    Call this after calling `annotate_ancestral_states`
    Assigns distance based on max parsimony
    @param do_fast: indicates whether to keep valid distances untouched and assume they are correct.
                    only recalculate distance for branches with negative distance
    @return parsimony score
    """
    # TODO: add python unit test for this fast version
    # The parsimony score is the sum of the number of times a new singleton is introduced
    # in the ancestral state list for each node
    pars_score = 0
    for node in tree.get_descendants("preorder"):
        if do_fast and node.dist >= 0:
            pars_score += node.dist
            continue

        branch_pars_score = 0
        for up_anc_state, node_anc_state in zip(node.up.anc_state_list, node.anc_state_list):
            up_sgs = set(up_anc_state.get_singletons())
            node_sgs = set(node_anc_state.get_singletons())
            branch_pars_score += len(node_sgs - up_sgs)
        node.dist = branch_pars_score
        pars_score += branch_pars_score
    return pars_score

def get_max_parsimony_anc_singletons(tree: CellLineageTree, bcode_meta: BarcodeMetadata):
    """
    Call this after calling `annotate_ancestral_states`
    @return a dict mapping node id to the max-parsimony ancestral state using the SGs in the anc state
    """
    max_parsimony_states = {}
    for node in tree.traverse("preorder"):
        max_parsimony_states[node.node_id] = []
        for idx, anc_state in enumerate(node.anc_state_list):
            node_sgs = anc_state.get_singletons()
            max_parsimony_states[node.node_id].append(node_sgs)
    return max_parsimony_states
