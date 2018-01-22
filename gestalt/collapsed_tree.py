from ete3 import TreeNode


class CollapsedTree:
    @staticmethod
    def collapse(raw_tree: TreeNode, deduplicate_sisters=False):
        tree = raw_tree.copy()
        tree.ladderize()
        # first remove non-pendant zero-length edges
        for node in tree.get_descendants(strategy='postorder'):
            if node.dist == 0 and not node.is_leaf():
                # TODO: one day we might want to think about collapsing only if the cell states are the same
                node.up.name = node.name
                node.delete(prevent_nondicotomic=False)
        # collapse identical sister leaves
        if deduplicate_sisters:
            done = False
            while not done:
                did_something = False
                for leaf in tree:
                    abundance = 1
                    for sister in leaf.get_sisters():
                        if sister.allele_events.events == leaf.allele_events.events:
                            abundance += 1
                            leaf.remove_sister(sister)
                            did_something = True
                    leaf.add_feature('abundance', abundance)
                for leaf in tree:
                    if len(leaf.get_sisters()) == 0:
                        leaf.up.delete(prevent_nondicotomic=False,
                                       preserve_branch_length=True)
                if not did_something:
                    done = True
        # no internal zero-length branches should remain
        internal_dists = [node.dist for node in tree.iter_descendants() if not node.is_leaf()]
        if len(internal_dists) > 0:
            assert min(internal_dists) > 0
        return tree
