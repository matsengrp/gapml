from ete3 import TreeNode
import numpy as np

class CollapsedTree:
    @staticmethod
    def collapse(
            raw_tree: TreeNode,
            collapse_zero_lens=False,
            collapse_zero_leaves=False,
            collapse_same_ancestral=False,
            collapse_first_appear=False):
        tree = raw_tree.copy()
        tree.ladderize()
        if collapse_zero_lens or collapse_zero_leaves:
            # first remove zero-length edges
            iterable_tree = tree.get_descendants(strategy='postorder') if collapse_zero_lens else tree
            for node in iterable_tree:
                if node.dist == 0:
                    # TODO: one day we might want to think about collapsing only if the cell states are the same
                    node.up.name = node.name
                    node.delete(prevent_nondicotomic=False)

        if collapse_same_ancestral:
            # collapse if ancestor node has exactly the same events
            for node in tree.get_descendants(strategy='postorder'):
                if not node.is_root() and node.allele_events.events == node.up.allele_events.events:
                    node.delete(prevent_nondicotomic=False, preserve_branch_length=True)

        # collapse to subtree to first appearance of each leaf
        if collapse_first_appear:
            did_something = True
            while did_something:
                did_something = False
                for leaf in tree:
                    if not leaf.is_root() and leaf.allele_events.events == leaf.up.allele_events.events:
                        # There is no branch length to preserve since this is a leaf...
                        leaf.delete(prevent_nondicotomic=False, preserve_branch_length=False)
                        did_something = True

            for node in tree.get_descendants(strategy="postorder"):
                if len(node.get_children()) == 1 and not node.is_root() and node.up.allele_events.events == node.allele_events.events:
                    node.delete(prevent_nondicotomic=True, preserve_branch_length=True)

        return tree
