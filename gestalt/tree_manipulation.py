import copy
import numpy as np

def NNI(node):
    """
    NNI move code copied from phyloinfer.
    assumes a bifurcating tree
    only performs NNI on interior branches
    """
    if node.is_root() or node.is_leaf():
        raise ValueError("Can not perform NNI on root or leaf branches!")
    else:
        neighboor = np.random.randint(2)
        xchild = node.get_sisters()[0]
        parent = node.up
        ychild = node.children[neighboor]
        if parent.allele_events_list_str == node.allele_events_list_str:
            # This is currently a jenky NNI check
            # Checks if oracle ancestor state is the same.
            # Then NNI move is allowed.
            # Preserves the parsimony score
            xchild.detach()
            ychild.detach()
            parent.add_child(ychild)
            node.add_child(xchild)
            return True
        else:
            return False

def search_nearby_trees(true_tree, max_search_dist=10):
    """
    Searches nearby trees, but makes sure that the gestalt rules
    are obeyed and parsimony score is preserved

    @param max_search_dist: perform this many NNI moves
    """
    # Make sure we don't change the original tree
    scratch_tree = copy.deepcopy(true_tree)

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
        nni_success = False
        while not nni_success:
            # Pick the random node
            rand_node_index = np.random.randint(0, num_interior_nodes)
            rand_node = node_dict[rand_node_index]
            # Perform the actual move
            nni_success = NNI(rand_node)

        # Make a copy of this tree and store!
        trees.append(copy.deepcopy(scratch_tree))
    return trees

