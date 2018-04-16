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
        xchild.detach()
        ychild.detach()
        parent.add_child(ychild)
        node.add_child(xchild)

def search_nearby_trees(true_tree, max_search_dist=10):
    """
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
        # Pick the random node
        rand_node_index = np.random.randint(0, num_interior_nodes)
        rand_node = node_dict[rand_node_index]
        # Perform the actual move
        NNI(rand_node)
        # Make a copy of this tree and store!
        trees.append(copy.deepcopy(scratch_tree))

    # Now calculate the rf distances of each random tree
    rf_tree_dict = {}
    for tree in trees:
        rf_res = true_tree.robinson_foulds(
                tree,
                attr_t1="allele_events_list_str",
                attr_t2="allele_events_list_str",
                expand_polytomies=False,
                unrooted_trees=False)
        print("rf dist", rf_res[0])
        rf_dist = rf_res[0]
        if rf_dist in rf_tree_dict:
            rf_tree_dict[rf_dist].append(tree)
        else:
            rf_tree_dict[rf_dist] = [tree]

    return rf_tree_dict

