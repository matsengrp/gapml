from phylip_parse import hamming_distance

class CollapsedTree:
    def __init__(self, raw_tree, preserve_leaves=False):
        tree = raw_tree.copy()
        for node in tree.get_descendants(strategy='postorder'):
            node.dist = hamming_distance(str(node.up.barcode), str(node.barcode))

        for node in tree.get_descendants(strategy='postorder'):
            if node.dist == 0 and (not node.is_leaf() or (node.is_leaf() and not preserve_leaves)):
                if node.up.name is "":
                    node.up.name = node.name
                else:
                    node.up.name += ",%s" % node.name
                node.delete(prevent_nondicotomic=False)

        for node in tree.get_descendants(strategy='preorder'):
            children = node.get_children()
            if len(children) == 1 and (not node.is_leaf() or (node.is_leaf() and not preserve_leaves)):
                node.delete(prevent_nondicotomic=True)
        self.tree = tree

