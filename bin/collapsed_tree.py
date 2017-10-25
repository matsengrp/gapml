from phylip_parse import hamming_distance

from cell_lineage_tree import CellLineageTree


class CollapsedTree:
    @staticmethod
    def collapse(raw_tree: CellLineageTree, preserve_leaves: bool=False):
        tree = raw_tree.copy()

        for node in tree.get_descendants(strategy='postorder'):
            node.dist = hamming_distance(
                str(node.up.barcode), str(node.barcode))
            if node.dist == 0 and (not node.is_leaf() or
                                   (node.is_leaf() and not preserve_leaves)):
                node.delete(prevent_nondicotomic=False)
        return tree

