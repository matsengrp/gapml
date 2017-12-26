from ete3 import TreeNode


class CollapsedTree:
    @staticmethod
    def collapse(raw_tree: TreeNode,
                 preserve_leaves: bool = True,
                 deduplicate_sisters: bool = True):
        tree = raw_tree.copy()
        tree.ladderize()
        for node in tree.get_descendants(strategy='postorder'):
            if node.dist == 0 and (not node.is_leaf() or not preserve_leaves):
                # TODO: one day we might want to think about collapsing only if the cell states are the same
                node.up.name = node.name
                node.delete(prevent_nondicotomic=False)
        if preserve_leaves and deduplicate_sisters:
            # collapse identical sister leaves
            for leaf in tree:
                abundance = 1
                for sister in leaf.get_sisters():
                    if sister.barcode_events.events == leaf.barcode_events.events:
                        abundance += 1
                        leaf.remove_sister(sister)
                leaf.add_feature('abundance', abundance)
            for leaf in tree:
                if len(leaf.get_sisters()) == 0:
                    leaf.up.delete(prevent_nondicotomic=False,
                                   preserve_branch_length=True)
            # no internal zero-length branches should remain
            assert min(node.dist for node in tree.iter_descendants() if not node.is_leaf()) > 0
        return tree
