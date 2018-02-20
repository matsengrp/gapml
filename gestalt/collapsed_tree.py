from ete3 import TreeNode
import numpy as np

class CollapsedTree:
    @staticmethod
    def collapse(
            raw_tree: TreeNode,
            collapse_zero_lens=False,
            deduplicate_sisters=False,
            deduplicate_parent_child=False):
        tree = raw_tree.copy()
        tree.ladderize()
        if collapse_zero_lens:
            # first remove non-pendant zero-length edges
            for node in tree.get_descendants(strategy='postorder'):
                if node.dist == 0 and not node.is_leaf():
                    # TODO: one day we might want to think about collapsing only if the cell states are the same
                    node.up.name = node.name
                    node.delete(prevent_nondicotomic=False)

        # collapse identical sister leaves
        if deduplicate_sisters:
            for leaf in tree:
                leaf.add_feature('dists', [leaf.dist])

            done = False
            while not done:
                did_something = False
                for leaf in tree:
                    dists = leaf.dists
                    for sister in leaf.get_sisters():
                        if sister.is_leaf() and sister.allele_events.events == leaf.allele_events.events:
                            dists.append(sister.dist)
                            leaf.remove_sister(sister)
                            did_something = True

                for leaf in tree:
                    if len(leaf.get_sisters()) == 0 and not leaf.is_root():
                        leaf.dist = np.mean(leaf.dists)
                        add_dist = leaf.up.dist
                        leaf.dists = [d + add_dist for d in leaf.dists]
                        leaf.up.delete(prevent_nondicotomic=False,
                                       preserve_branch_length=True)

                if not did_something:
                    done = True

            # collapse identical parent child leaves
            if deduplicate_parent_child:
                # But make sure we don't collapse away all of the observed leaves
                # we keep a count of the unique observed leaves.
                # after collapsing, we want to make sure we still have one observed leaf
                obs_count = {}
                for leaf in tree:
                    leaf_evts = tuple(leaf.allele_events.events)
                    if leaf_evts not in obs_count:
                        obs_count[leaf_evts] = 1
                    else:
                        obs_count[leaf_evts] += 1

                for leaf in tree:
                    dists = leaf.dists
                    leaf_evts = tuple(leaf.allele_events.events)
                    if leaf.allele_events.events == leaf.up.allele_events.events and obs_count[leaf_evts] > 1:
                        leaf.delete(preserve_branch_length=True)
                        obs_count[leaf_evts] -= 1

        return tree
