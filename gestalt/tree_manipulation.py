import copy
import typing
import numpy as np

from indel_sets import SingletonWC
import ancestral_events_finder as anc_evt_finder
from cell_lineage_tree import CellLineageTree

"""
TODO: add types to the arguments!
"""

def _resolve_multifurc(node: CellLineageTree, c_index: int):
    """
    Resolve the multifurcation at this node by adding a bifurcation
    with child `c_index` as the first one to split off
    @param node: node to resolve multifurcation for
    @param c_index: the child to peel off

    @return whether or not we successfully resolve this multifurcation
    """
    node_str = node.allele_events_list_str
    # reconstruct the tree
    child = node.get_children()[c_index]
    if child.allele_events_list_str == node_str:
        # Cannot resolve if the request was to set the parent ancestral allele to a child
        return False

    new_inner_node = CellLineageTree(
            allele_list=node.allele_list,
            allele_events_list=node.allele_events_list,
            cell_state=node.cell_state)
    new_inner_node.add_feature("observed", node.observed)

    sisters = child.get_sisters()
    for s in sisters:
        s.detach()
        new_inner_node.add_child(s)
    node.add_child(new_inner_node)
    return True

def resolve_all_multifurcs(tree: CellLineageTree):
    """
    resolve all the multifurcations in this tree
    """
    no_multifurcs = False
    while not no_multifurcs:
        # TODO: worlds most ineffecieint resolver. oh well
        no_multifurcs = True
        for node in tree.traverse():
            children = node.get_children()
            num_children = len(children)
            if num_children > 2:
                # This is a multifurc
                rand_child_idx = np.random.randint(low=0, high=num_children)
                did_rearrange = _resolve_multifurc(node, rand_child_idx)
                no_multifurcs = False
                break

def NNI(node: CellLineageTree, xchild: CellLineageTree, ychild: CellLineageTree):
    """
    NNI move code copied from phyloinfer.
    assumes a bifurcating tree
    only performs NNI on interior branches

    @param node: the node to do NNI around
    @param xchild: sister of `node`. swap with ychild
    @param ychild: child of `node`. swap with xchild
    """
    if node.is_root() or node.is_leaf():
        raise ValueError("Can not perform NNI on root or leaf branches!")
    else:
        x_dist = xchild.dist
        parent = node.up
        y_dist = ychild.dist
        # Preserves the parsimony score
        xchild.detach()
        ychild.detach()
        parent.add_child(ychild)
        # preserve ultrametric by swapping branch lengths
        ychild.dist = x_dist
        node.add_child(xchild)
        xchild.dist = y_dist

        # Redo ancestral state calculations because of NNI moves
        # TODO: reset the strings that are printed too.
        node.anc_state_list = anc_evt_finder.get_possible_anc_states(node)
        node.up.anc_state_list = anc_evt_finder.get_possible_anc_states(node.up)

def get_parsimony_edge_score(node: CellLineageTree):
    """
    Get the parsimony score for that edge
    """
    score = 0
    for anc_state, par_anc_state in zip(node.anc_state_list, node.up.anc_state_list):
        node_singletons = set(anc_state.get_singleton_wcs())
        par_singletons = set(par_anc_state.get_singleton_wcs())
        score += len(node_singletons - par_singletons)
    return score

def get_parsimony_score_nearest_neighbors(node: CellLineageTree):
    """
    Get the parsimony score summing over this node, its children, and its sisters.
    """
    score = get_parsimony_edge_score(node)
    for child in node.children:
        score += get_parsimony_edge_score(child)
    for sister in node.get_sisters():
        score += get_parsimony_edge_score(sister)
    return score

def search_nearby_trees(init_tree: CellLineageTree, max_search_dist: int =10):
    """
    Searches nearby trees, but makes sure that the gestalt rules
    are obeyed and parsimony score is preserved

    @param max_search_dist: perform this many NNI moves
    """
    # Make sure we don't change the original tree
    scratch_tree = init_tree.copy("deepcopy")

    # First label the nodes in the tree
    node_dict = []
    for node in scratch_tree.traverse():
        if node.is_root() or node.is_leaf():
            continue
        node_dict.append(node)
    num_interior_nodes = len(node_dict)

    # Now perform NNI moves on random nodes
    curr_tree = scratch_tree
    trees = []
    for i in range(max_search_dist):
        while True:
            # Pick the random node
            rand_node_index = np.random.randint(0, num_interior_nodes)
            rand_node = node_dict[rand_node_index]
            if rand_node.is_leaf() or rand_node.is_root():
                continue
            else:
                # Propose an NNI move
                sisters = rand_node.get_sisters()
                sister_idx = np.random.randint(len(sisters))
                xchild = sisters[sister_idx]
                if get_parsimony_edge_score(xchild) == 0:
                    if len(sisters) == 1:
                        xchild = rand_node
                        rand_node = sisters[sister_idx]
                        if rand_node.is_leaf() or rand_node.is_root():
                            continue
                    else:
                        while get_parsimony_edge_score(xchild) == 0:
                            sister_idx = np.random.randint(len(sisters))
                            xchild = sisters[sister_idx]
                assert get_parsimony_edge_score(xchild) > 0

                children = rand_node.get_children()
                child_idx = np.random.randint(len(children))
                ychild = rand_node.children[child_idx]
                while get_parsimony_edge_score(ychild) == 0:
                    child_idx = np.random.randint(len(children))
                    ychild = rand_node.children[child_idx]

                # Get the orig parsimony score of this region
                orig_score = get_parsimony_score_nearest_neighbors(rand_node)

                # Do the NNI
                NNI(rand_node, xchild, ychild)
                # Calculate the new parismony score of this region
                proposed_score = get_parsimony_score_nearest_neighbors(rand_node)
                if proposed_score > orig_score:
                    # If we increase the parsimony score, then undo the NNI
                    # This just requires swapping the inputs to the NNI function
                    NNI(rand_node, ychild, xchild)
                else:
                    break

        # Make a copy of this tree and store!
        trees.append(scratch_tree.copy("deepcopy"))
    return trees
