from ete3 import TreeNode
import numpy as np

class CollapsedTree:
    @staticmethod
    def collapse(
            raw_tree: TreeNode,
            collapse_zero_lens=False,
            deduplicate=False):
        tree = raw_tree.copy()
        tree.ladderize()
        if collapse_zero_lens:
            # first remove zero-length edges
            #for node in tree.get_descendants(strategy='postorder'):
            #    if node.dist == 0 and node.is_leaf():
            for node in tree:
                if node.dist == 0:
                    # TODO: one day we might want to think about collapsing only if the cell states are the same
                    node.up.name = node.name
                    node.delete(prevent_nondicotomic=False)

        # collapse identical nodes
        if deduplicate:
            did_something = True
            while did_something:
                did_something = False
                for leaf in tree:
                    if leaf.allele_events.events == leaf.up.allele_events.events:
                        leaf.delete(prevent_nondicotomic=False, preserve_branch_length=True)
                        did_something = True

            for node in tree.get_descendants(strategy="postorder"):
                if len(node.get_children()) == 1 and not node.is_root() and node.up.allele_events.events == node.allele_events.events:
                    node.delete(prevent_nondicotomic=True, preserve_branch_length=True)

        return tree
