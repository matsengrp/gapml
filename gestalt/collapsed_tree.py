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

    print(tree.get_ascii(attributes=["dist"], show_internal=True))
    print(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

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
    latest_allele_node_dict = {tree.allele_events_list_str: (collapsed_tree, 0)}
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
            parent_collapsed_node.add_child(new_child_collapsed_node)
            node_to_collapsed_node_dict[child.node_id] = new_child_collapsed_node
            latest_allele_node_dict[child.allele_events_list_str] = (new_child_collapsed_node, child.dist_to_root)
        else:
            # If the child allele same as parent, attach the new collapsed child
            # to the latest node with that same allele
            parent_collapsed_node, parent_dist_to_root = latest_allele_node_dict[parent.allele_events_list_str]
            collapsed_child_dist = child.dist_to_root - parent_dist_to_root
            new_child_collapsed_node = CellLineageTree(
                allele_list = child.allele_list,
                allele_events_list = child.allele_events_list,
                cell_state = child.cell_state,
                dist = collapsed_child_dist)
            parent_collapsed_node.add_child(new_child_collapsed_node)
            node_to_collapsed_node_dict[child.node_id] = new_child_collapsed_node
            latest_allele_node_dict[child.allele_events_list_str] = (new_child_collapsed_node, child.dist_to_root)

    # Mark nodes as observed or not
    for node in collapsed_tree.traverse():
        # Any node that has distance to root equal to max dist is observed
        node.add_feature("observed", node.is_leaf())

    _remove_single_child_unobs_nodes(collapsed_tree)

    print(collapsed_tree.get_ascii(attributes=["dist"], show_internal=True))
    print(collapsed_tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    print(collapsed_tree.get_ascii(attributes=["observed"], show_internal=True))
    for leaf in tree:
        assert np.isclose(leaf.get_distance(tree), max_dist)

    return collapsed_tree

def collapse_zero_lens(raw_tree: TreeNode):
    """
    Remove zero-length edges from the tree, but leave one leaf node for each observed node.
    """
    tree = _preprocess(raw_tree)
    # All nodes must have a name!!!
    assert tree.name != ""
    # dictionary maps node name to a node that was removed because it was a "duplicate"
    # (i.e. zero length away). We use this dictionary later to add nodes back in
    removed_children_dict = {}

    # Step 1: just collapse all zero length edges. So if an observed node was a parent
    # of another node, it will no longer be a leaf node after this procedure.
    for node in tree.traverse(strategy='postorder'):
        if not hasattr(node, "observed"):
            # All leaf nodes are observed
            node.add_feature("observed", node.is_leaf())

        if node.dist == 0 and not node.is_root():
            # Need to remove this node and propogate the observed status
            # up to the parent node (if parent observed already, dont update
            # the parent)
            up_node = node.up
            if hasattr(up_node, "observed"):
                if not up_node.observed:
                    up_node.add_feature("observed", node.observed)
                    up_node.name = node.name
            else:
                up_node.add_feature("observed", node.observed)
                up_node.name = node.name
            node.delete(prevent_nondicotomic=False)
            if up_node.name not in removed_children_dict:
                removed_children_dict[up_node.name] = node

    # Step 2: Clean up the tree so that all nodes have at least two children
    _remove_single_child_unobs_nodes(tree)

    # Step 3: add nodes that were observed but were collapsed away.
    # This ensures only one leaf node for each observed allele.
    for node in tree.traverse("preorder"):
        node.add_feature("is_copy", False)

    for node in tree.traverse("preorder"):
        if node.observed and not node.is_leaf():
            # Use a node from the dictionary as a template
            child_template = removed_children_dict[node.name]
            # Copy this node and its features... don't use the original one just in case
            if tree.__class__ == CellLineageTree:
                new_child = CellLineageTree.convert(
                    child_template,
                    allele_list = child_template.allele_list,
                    allele_events_list = child_template.allele_events_list,
                    cell_state = child_template.cell_state,
                    resolved_multifurcation = False)
            else:
                new_child = TreeNode(
                        name=child_template.name,
                        dist=child_template.dist)
                for k in child_template.features:
                    if k not in new_child.features:
                        new_child.add_feature(k, getattr(child_template, k))
            node.add_child(new_child)
            new_child.add_feature("is_copy", True)
            new_child.observed = True
            node.observed = False

    return tree
