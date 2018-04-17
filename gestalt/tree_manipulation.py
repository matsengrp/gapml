import copy
import numpy as np

from indel_sets import SingletonWC
import ancestral_events_finder as anc_evt_finder

def NNI(node, xchild, ychild):
    """
    NNI move code copied from phyloinfer.
    assumes a bifurcating tree
    only performs NNI on interior branches
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

def get_parsimony_edge_score(node):
    """
    Get the parsimony score for that edge
    """
    score = 0
    for anc_state, par_anc_state in zip(node.anc_state_list, node.up.anc_state_list):
        node_singletons = set(anc_state.get_singleton_wcs())
        par_singletons = set(par_anc_state.get_singleton_wcs())
        score += len(node_singletons - par_singletons)
    return score

def get_parsimony_score_nearest_neighbors(node, redo_ancestral_states=True):
    """
    Get the parsimony score summing over this node, its children, and its sisters.
    """
    if redo_ancestral_states:
        node.anc_state_list = anc_evt_finder.get_possible_anc_states(node)
        node.up.anc_state_list = anc_evt_finder.get_possible_anc_states(node.up)

    score = get_parsimony_edge_score(node)
    for child in node.children:
        score += get_parsimony_edge_score(child)
    for sister in node.get_sisters():
        score += get_parsimony_edge_score(sister)
    return score

def search_nearby_trees(init_tree, max_search_dist=10):
    """
    Searches nearby trees, but makes sure that the gestalt rules
    are obeyed and parsimony score is preserved

    @param max_search_dist: perform this many NNI moves
    """
    # Make sure we don't change the original tree
    scratch_tree = copy.deepcopy(init_tree)

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

                children = rand_node.get_children()
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
        trees.append(copy.deepcopy(scratch_tree))
    return trees

