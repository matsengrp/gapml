from ete3 import Tree #NodeStyle, TreeStyle, SeqMotifFace

from phylip_parse import hamming_distance


class BarcodeTree():
    """
    Each node has a barcode feature
    """
    def __init__(self, tree: Tree):
        self.tree = tree

    def render(self, file):
        '''render tree to image file'''
        style = NodeStyle()
        style['size'] = 0
        for n in self.tree.traverse():
           n.set_style(style)
        for leaf in self.tree:
           seqFace = SeqMotifFace(seq=str(leaf.barcode).upper(),
                                  seqtype='nt',  # nucleotide seq
                                  seq_format='[]',
                                  height=3,
                                  gapcolor='red',
                                  fgcolor='black',
                                  bgcolor='lightgray',
                                  width=5)
           leaf.add_face(seqFace, 1)
        tree_style = TreeStyle()
        tree_style.show_scale = False
        tree_style.show_leaf_name = False
        self.tree.render(file, tree_style=tree_style)

    def n_leaves(self):
        return len(self.tree)


class CollapsedTree:
    def __init__(self, raw_tree, preserve_leaves=False):
        tree = raw_tree.copy()

        for node in tree.get_descendants(strategy='preorder'):
            node.dist = hamming_distance(str(node.up.barcode), str(node.barcode))
            if node.dist == 0 and (not node.is_leaf() or (node.is_leaf() and not preserve_leaves)):
                node.delete(prevent_nondicotomic=True)

        self.tree = tree
