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

def _get_earliest_uniq_branches(tree: TreeNode):
    """
    @return a dict with all the unique pairs of parent to child alleles in the tree.
            The dict maps child allele string to [
                the earliest node with the child allele,
                the distance from this earliest node to the root,
                the parent node's allele string,
                whether or not the allele is observed]
    """
    uniq_allele_branches = dict()
    for node in tree.traverse():
        allele_id = node.allele_events_list_str
        node_up_dist = 0 if node.is_root() else node.up.dist_to_root
        if allele_id in uniq_allele_branches:
            shortest_dist = uniq_allele_branches[allele_id][1]
            is_observed = uniq_allele_branches[allele_id][-1]
            if node.up.dist_to_root < shortest_dist:
                uniq_allele_branches[allele_id] = [
                        node,
                        node_up_dist,
                        node.up,
                        is_observed]
        else:
            uniq_allele_branches[allele_id] = [
                    node,
                    node_up_dist,
                    node.up,
                    False]
        # An allele is observed if the allele ever appears at a leaf
        uniq_allele_branches[allele_id][-1] |= node.is_leaf()
    return uniq_allele_branches

def _regrow_collapsed_tree(tree: TreeNode, uniq_allele_branches: Dict):
    """
    Regrow the tree per the rules about first appearance.
    Modifies `tree` to create the new tree with `tree` as the root

    @return a dictionary mapping
        allele => a dictionary mapping
                        dist to root => node with that allele
    """
    # Sort the branches/alleles so that we process branches sequentially by the time of creation/
    # the time these alleles branched off.
    sorted_branches = sorted(list(uniq_allele_branches.values()), key=lambda br: br[1])
    # Useful dictionary to tracking what nodes we have created so far
    existing_nodes_dict = {tree.allele_events_list_str: {0: tree}}
    for node, dist_to_root, old_parent_node, is_obs in sorted_branches:
        # Check if this node is not the root
        if node.allele_events_list_str != NO_EVT_STR:
            parent_allele_str = old_parent_node.allele_events_list_str
            print(parent_allele_str, "=>", node.allele_events_list_str, dist_to_root)
            if parent_allele_str in existing_nodes_dict:
                if dist_to_root in existing_nodes_dict[parent_allele_str]:
                    # There already exists a node with the parent allele that is exactly this distance
                    # away from the root.
                    # This seems like it would happen only in rare cases?
                    par_node = existing_nodes_dict[parent_allele_str][dist_to_root]
                else:
                    # There does not exist a node with this parent allele of this distance
                    # away from the root, though there are nodes with this parent allele
                    # that are closer to the root. We need to make this new parent node and
                    # connect this new node to its appropriate parent node
                    max_dist_to_root = max(list(existing_nodes_dict[parent_allele_str].keys()))
                    grandpa_node = existing_nodes_dict[parent_allele_str][max_dist_to_root]
                    assert dist_to_root > max_dist_to_root
                    # Use the grandpa node as a template for creating a new parent node
                    par_node = CellLineageTree(
                            allele_list = old_parent_node.allele_list,
                            cell_state = old_parent_node.cell_state,
                            dist = dist_to_root - max_dist_to_root)
                    par_node.add_feature("dist_to_root", dist_to_root)
                    grandpa_node.add_child(par_node)
                    existing_nodes_dict[parent_allele_str][dist_to_root] = par_node
            else:
                raise ValueError("Should never happen -- allele should already exist")

            if node.allele_events_list_str in existing_nodes_dict:
                raise ValueError("Should never happen -- allele cant exist already")
            else:
                # Now put this parent node in this new allele's dictionary -- it is the first
                # timepoint where the new allele is branching off of the parent allele.
                # Now we can create children nodes that branch off of this first timepoint.
                existing_nodes_dict[node.allele_events_list_str] = {
                        dist_to_root: par_node}
    return existing_nodes_dict

def _remove_single_child_unobs_nodes(tree: TreeNode):
    """
    Remove single link children from the root node until there is at most two single links
    """
    while len(tree.get_children()) == 1 and len(tree.get_children()[0].get_children()) == 1:
        child_node = tree.get_children()[0]
        # Preserve branch lengths by propagating down (ete does this wrong)
        child_node_dist = child_node.dist
        for grandchild in child_node.children:
            grandchild.dist += child_node_dist
        child_node.delete(prevent_nondicotomic=True, preserve_branch_length=False)
    assert(tree.is_root())

    for node in tree.get_descendants(strategy="postorder"):
        if len(node.get_children()) == 1 and not node.observed:
            node.delete(prevent_nondicotomic=True, preserve_branch_length=True)

def collapse_ultrametric(raw_tree: TreeNode):
    tree = _preprocess(raw_tree)
    tree.label_tree_with_strs()

    max_dist = _label_dist_to_root(tree)

    uniq_allele_branches = _get_earliest_uniq_branches(tree)

    # We break up the whole tree
    for node in tree.traverse(strategy="postorder"):
        node.detach()

    # We are ready to reconstruct the collapsed ultrametric tree
    existing_nodes_dict = _regrow_collapsed_tree(tree, uniq_allele_branches)

    # Now create leaf nodes that ensure we satisfy the ultrametric assumption
    for node, _, _, is_observed in uniq_allele_branches.values():
        allele_id = node.allele_events_list_str
        if is_observed and max_dist not in existing_nodes_dict[allele_id]:
            max_dist_to_root = max(list(existing_nodes_dict[allele_id].keys()))
            par_node = existing_nodes_dict[allele_id][max_dist_to_root]
            new_node = CellLineageTree(
                    allele_list = node.allele_list,
                    cell_state = node.cell_state,
                    dist = max_dist - max_dist_to_root)
            new_node.add_feature("dist_to_root", max_dist)
            par_node.add_child(new_node)

    # Mark nodes as observed or not
    for node in tree.traverse():
        # Any node that has distance to root equal to max dist is observed
        node.add_feature("observed", node.dist_to_root == max_dist)

    _remove_single_child_unobs_nodes(tree)

    print(tree.get_ascii(attributes=["dist"], show_internal=True))
    print(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
    print(tree.get_ascii(attributes=["observed"], show_internal=True))
    for leaf in tree:
        assert np.isclose(leaf.get_distance(tree), max_dist)

    return tree

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
