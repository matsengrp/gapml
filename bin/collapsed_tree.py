from ete3 import TreeNode


class CollapsedTree:
    @staticmethod
    def collapse(raw_tree: TreeNode, preserve_leaves: bool = False):
        tree = raw_tree.copy()

        for node in tree.get_descendants(strategy='postorder'):
            if node.dist == 0 and (not node.is_leaf() or
                                   (node.is_leaf() and not preserve_leaves)):
                # TODO: one day we might want to think about collapsing only if the cell states are the same
                node.delete(prevent_nondicotomic=False)
        return tree
