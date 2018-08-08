from typing import List, Dict
import logging

from ete3 import TreeNode
import numpy as np

from cell_lineage_tree import CellLineageTree
from constants import NO_EVT_STR

def _preprocess(raw_tree:TreeNode):
    tree = raw_tree.copy()
    tree.ladderize()
    return tree

def _label_dist_to_root(tree: TreeNode):
    """
    labels each node with distance to root
    @return maximum distance to root
    """
    for node in tree.traverse():
        node.add_feature("dist_to_root", node.get_distance(tree))
        if node.is_leaf():
            max_dist = node.dist_to_root
    return max_dist

def _remove_single_child_unobs_nodes(tree: TreeNode):
    """
    Remove single link children from the root node until there is at most two single links
    """
    while len(tree.get_children()) == 1 and len(tree.get_children()[0].get_children()) == 1:
        child_node = tree.get_children()[0]
        grandchild_node = child_node.get_children()[0]
        # Preserve branch lengths by propagating down (ete does this wrong)
        child_node_dist = child_node.dist
        grandchild_node.dist += child_node_dist
        child_node.delete(prevent_nondicotomic=True, preserve_branch_length=False)

    for node in tree.get_descendants(strategy="postorder"):
        if len(node.get_children()) == 1:
            node.delete(prevent_nondicotomic=True, preserve_branch_length=True)

def collapse_ultrametric(raw_tree: CellLineageTree):
    """
    @return a collapsed CellLineageTree like that in the manuscript (the collapsed tree
            from the filtration that cannot distinguish between nodes with the same alleles)
    """
    tree = _preprocess(raw_tree)
    tree.label_tree_with_strs()
    tree.label_node_ids()
    max_dist = _label_dist_to_root(tree)

    #print(tree.get_ascii(attributes=["dist"], show_internal=True))
    #print(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    #print(tree.get_ascii(attributes=["node_id"], show_internal=True))

    tree.add_feature("earliest_ancestor_node_id", tree.node_id)
    for node in tree.iter_descendants():
        if node.up.allele_events_list_str == node.allele_events_list_str:
            node.add_feature("earliest_ancestor_node_id", node.up.earliest_ancestor_node_id)
        else:
            node.add_feature("earliest_ancestor_node_id", node.node_id)

    # Get all the branches in the original tree and sort them by time
    branches = []
    for node in tree.iter_descendants():
        branch = {
                "parent": node.up,
                "child": node,
                "dist_to_root": node.dist_to_root}
        branches.append(branch)
    sorted_branches = sorted(branches, key=lambda x: x["dist_to_root"])

    # Begin creating the collapsed tree
    collapsed_tree = CellLineageTree(
            allele_list = tree.allele_list,
            allele_events_list = tree.allele_events_list,
            cell_state = tree.cell_state)
    node_to_collapsed_node_dict = {tree.node_id: collapsed_tree}
    latest_node_dict = {tree.earliest_ancestor_node_id: (collapsed_tree, 0)}
    for branch in sorted_branches:
        child = branch["child"]
        parent = branch["parent"]

        is_diff_allele = child.allele_events_list_str != parent.allele_events_list_str
        if is_diff_allele:
            # If child allele differs from parent, attach the new collapsed child to the matching
            # collapsed parent
            parent_collapsed_node = node_to_collapsed_node_dict[parent.node_id]
            new_child_collapsed_node = CellLineageTree(
                allele_list = child.allele_list,
                allele_events_list = child.allele_events_list,
                cell_state = child.cell_state,
                dist = child.dist)
        else:
            # If the child allele same as parent, attach the new collapsed child
            # to the latest node with that same allele
            parent_collapsed_node, parent_dist_to_root = latest_node_dict[child.earliest_ancestor_node_id]
            abund = 1
            collapsed_child_dist = child.dist_to_root - parent_dist_to_root
            if np.isclose(0, collapsed_child_dist):
                abund = parent_collapsed_node.abundance + 1
            new_child_collapsed_node = CellLineageTree(
                allele_list = child.allele_list,
                allele_events_list = child.allele_events_list,
                cell_state = child.cell_state,
                dist = collapsed_child_dist,
                abundance = abund)
        parent_collapsed_node.add_child(new_child_collapsed_node)
        node_to_collapsed_node_dict[child.node_id] = new_child_collapsed_node
        latest_node_dict[child.earliest_ancestor_node_id] = (new_child_collapsed_node, child.dist_to_root)

    for leaf in collapsed_tree:
        assert np.isclose(leaf.get_distance(collapsed_tree), max_dist)

    _remove_single_child_unobs_nodes(collapsed_tree)

    #print(collapsed_tree.get_ascii(attributes=["dist"], show_internal=True))
    #print(collapsed_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

    return collapsed_tree

def collapse_zero_lens(raw_tree: TreeNode):
    """
    Remove zero-length edges from the tree, but leave one leaf node for each observed node.
    """
    tree = _preprocess(raw_tree)

    for node in tree.traverse(strategy='postorder'):
        if node.dist == 0 and not node.is_root() and not node.is_leaf():
            node.delete(prevent_nondicotomic=False)

    _remove_single_child_unobs_nodes(tree)

    for node in tree.traverse("preorder"):
        node.add_feature("observed", node.is_leaf())

    return tree
