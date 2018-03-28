from ete3 import TreeNode
import numpy as np
from typing import List
import logging


class CollapsedTree:
    @staticmethod
    def _preprocess(raw_tree:TreeNode):
        tree = raw_tree.copy()
        tree.ladderize()
        return tree

    @staticmethod
    def collapse_zero_lens(raw_tree: TreeNode, feature_name: str = "observed"):
        1/0
        tree = CollapsedTree._preprocess(raw_tree)
        # remove zero-length edges
        for node in tree.traverse(strategy='postorder'):
            if not hasattr(node, feature_name):
                node.add_feature(feature_name, node.is_leaf())
            if node.dist == 0 and not node.is_root():
                # TODO: one day we might want to think about collapsing only if the cell states are the same
                up_node = node.up
                up_node.name = node.name
                node.delete(prevent_nondicotomic=False)
                up_node.add_feature(feature_name, getattr(node, feature_name))

        for node in tree.get_descendants(strategy="postorder"):
            if len(node.get_children()) == 1 and not getattr(node, feature_name):
                logging.info("WARNING: There was an inner node with only one child!")
                node.delete(prevent_nondicotomic=True, preserve_branch_length=True)

        return tree

    @staticmethod
    def collapse_zero_leaves(raw_tree: TreeNode):
        1/0
        tree = CollapsedTree._preprocess(raw_tree)
        # remove zero-length edges to leaves
        for node in tree:
            if node.dist == 0:
                # TODO: one day we might want to think about collapsing only if the cell states are the same
                node.up.name = node.name
                node.delete(prevent_nondicotomic=False)
        return tree

    @staticmethod
    def collapse_same_ancestral(raw_tree: TreeNode, feature_name: str = "observed", idxs: List[int] = None):
        """
        @param feature_name: This feature will be attached to each node
                            The leaf nodes will have this feature be true
                            When collapsing leaf nodes into the tree, the upper node will
                            keep this feature. This is useful for understanding which nodes
                            were collapsed from the leaves.
        """
        1/0
        tree = CollapsedTree._preprocess(raw_tree)
        tree.label_tree_with_strs()

        if idxs is None:
            idxs = range(len(tree.allele_events_list))

        # collapse if ancestor node has exactly the same events
        for node in tree.traverse(strategy='postorder'):
            if node.is_leaf():
                node.add_feature(feature_name, True)
            else:
                is_observed = False
                for c in node.get_children():
                    all_same = True
                    if len(c.allele_events_list) != len(node.allele_events_list):
                        all_same = False
                        continue
                    for idx in idxs:
                        c_evts = c.allele_events_list[idx]
                        node_evts = node.allele_events_list[idx]
                        if not c_evts == node_evts:
                            all_same = False
                            break
                    # Then everything is the same
                    if all_same:
                        c.delete(prevent_nondicotomic=False, preserve_branch_length=True)
                        is_observed |= getattr(c, feature_name)
                node.add_feature(feature_name, is_observed)

        for node in tree.get_descendants(strategy="postorder"):
            if len(node.get_children()) == 1 and not getattr(node, feature_name):
                up_node = node.up
                node.delete(prevent_nondicotomic=False, preserve_branch_length=True)

        return tree

    @staticmethod
    def collapse_identical_leaves(raw_tree: TreeNode, feature_name: str = "observed", idxs: List[int] = None):
        1/0
        tree = CollapsedTree._preprocess(raw_tree)
        observed_alleles = set()
        for leaf in tree:
            leaf_evts_tuple = tuple(leaf.allele_events_list)
            if leaf_evts_tuple in observed_alleles:
                logging.info("Spontaneous leaf found %s", leaf.allele_events_list_str)
                leaf.delete(prevent_nondicotomic=False)
                anc = leaf.up
                while not anc.is_root():
                    any_obs = False
                    for node_of_anc in anc.traverse():
                        if getattr(leaves_anc, feature_name):
                            any_obs = True
                            break
                    if any_obs:
                        # This has some observed children, so stop going up tree.
                        break
                    else:
                        # This anc node has no observed children, so keep going up tree.
                        anc.detach()
                        anc = leaf.up
            else:
                observed_alleles.add(leaf_evts_tuple)
        return tree

    @staticmethod
    def collapse_ultrametric(raw_tree: TreeNode):
        from cell_lineage_tree import CellLineageTree
        tree = CollapsedTree._preprocess(raw_tree)
        tree.label_tree_with_strs()

        observed_allele_strs = set()
        for node in tree.traverse():
            if node.is_root():
                node.add_feature("dist_to_root", 0)
            else:
                node.add_feature("dist_to_root", node.dist + node.up.dist_to_root)
            if node.is_leaf():
                observed_allele_strs.add(node.allele_events_list_str)
                max_dist = node.dist_to_root

        print("BEFOREEE")
        print(tree.get_ascii(attributes=["dist"], show_internal=True))
        print(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        print("BEFOREEE")

        uniq_allele_branches = dict()
        for node in tree.traverse():
            allele_id = node.allele_events_list_str
            allele_up_id = None if node.is_root() else node.up.allele_events_list_str
            node_up_dist = 0 if node.is_root() else node.up.dist_to_root
            if allele_id in uniq_allele_branches:
                shortest_dist = uniq_allele_branches[allele_id][1]
                is_observed = uniq_allele_branches[allele_id][-1]
                if node.up.dist_to_root < shortest_dist:
                    uniq_allele_branches[allele_id] = [
                            node,
                            node_up_dist,
                            allele_up_id,
                            is_observed]
            else:
                uniq_allele_branches[allele_id] = [
                        node,
                        node_up_dist,
                        allele_up_id,
                        False]
            uniq_allele_branches[allele_id][-1] = uniq_allele_branches[allele_id][-1] or node.is_leaf()

        for k, (node, dist_to_root, parent_allele_str, is_obs) in uniq_allele_branches.items():
            print(node.allele_events_list_str, dist_to_root, parent_allele_str, is_obs)

        no_events_str = tree.allele_events_list_str
        for node in tree.traverse(strategy="postorder"):
            node.detach()

        sorted_branches = sorted(list(uniq_allele_branches.values()), key=lambda br: br[1])
        existing_nodes_dict = {
                tree.allele_events_list_str: {0: tree}
            }
        tree.add_feature('observed', False)
        for node, dist_to_root, parent_allele_str, is_obs in sorted_branches:
            print(parent_allele_str, "=>", node.allele_events_list_str, dist_to_root)
            if node.allele_events_list_str != no_events_str:
                # Check if this node is not the root
                if parent_allele_str in existing_nodes_dict:
                    if dist_to_root in existing_nodes_dict[parent_allele_str]:
                        par_node = existing_nodes_dict[parent_allele_str][dist_to_root]
                        print("a??")
                    else:
                        max_dist_to_root = max(list(existing_nodes_dict[parent_allele_str].keys()))
                        grandpa_node = existing_nodes_dict[parent_allele_str][max_dist_to_root]
                        print("GGGG", max_dist_to_root, grandpa_node.allele_events_list_str)
                        assert dist_to_root > max_dist_to_root
                        print(dist_to_root, max_dist, dist_to_root == max_dist)
                        if dist_to_root == max_dist:
                            par_node_old = node
                        else:
                            par_node_old = uniq_allele_branches[parent_allele_str][0]
                        par_node = CellLineageTree(
                                allele_list = par_node_old.allele_list,
                                cell_state = par_node_old.cell_state,
                                dist = dist_to_root - max_dist_to_root)
                        par_node.add_feature("dist_to_root", dist_to_root)
                        grandpa_node.add_child(par_node)
                        existing_nodes_dict[parent_allele_str][dist_to_root] = par_node
                        print("yay", par_node.allele_events_list_str)
                else:
                    print(existing_nodes_dict.keys())
                    raise ValueError("huh?")
                par_node.add_feature("observed", dist_to_root == max_dist)
                
                if node.allele_events_list_str in existing_nodes_dict:
                    print("DERPRPRPP")
                    raise ValueError("huh????")
                    existing_nodes_dict[node.allele_events_list_str][dist_to_root] = par_node
                else:
                    print("noddd", node.allele_events_list_str)
                    existing_nodes_dict[node.allele_events_list_str] = {
                            dist_to_root: par_node}

        print(tree.get_ascii(attributes=["dist"], show_internal=True))
        print(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))

        for k, (node, dist_to_root, parent_allele_str, is_obs) in uniq_allele_branches.items():
            allele_id = node.allele_events_list_str
            if is_obs and max_dist not in existing_nodes_dict[allele_id]:
                print("need to make leaves", allele_id)
                max_dist_to_root = max(list(existing_nodes_dict[allele_id].keys()))
                par_node = existing_nodes_dict[allele_id][max_dist_to_root]
                new_node = CellLineageTree(
                        allele_list = node.allele_list,
                        cell_state = node.cell_state,
                        dist = max_dist - max_dist_to_root)
                new_node.add_feature("dist_to_root", max_dist)
                new_node.add_feature("observed", True)
                par_node.add_child(new_node)

        print(tree.get_ascii(attributes=["dist"], show_internal=True))
        print(tree.get_ascii(attributes=["allele_events_list_str"], show_internal=True))
        print(tree)
        return tree
