
from ete3 import TreeNode, NodeStyle, TreeStyle, faces, SeqMotifFace, add_face_to_node

from cell_state import CellType

class CellTypeTree:
    def __init__(self, cell_type: CellType=None):
        """
        @param cell_type: the cell type of this node is the union of labeled cell types of the
                            descendant nodes (including this node)
        """
        self.tree = TreeNode()
        self.tree.add_feature("cell_type", cell_type)


