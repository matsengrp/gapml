from ete3 import TreeNode

from barcode import Barcode
from cell_state import CellState

class CellLineageTree(TreeNode):
    """
    History from embryo cell to observed cells. Each node represents a cell divison/death.
    Class can be used for storing information about the true cell lineage tree and can be
    used for storing the estimate of the cell lineage tree.
    """
    def __init__(self, barcode: Barcode, cell_state: CellState, dist: float=0, dead=False):
        """
        @param dist: branch length from parent node
        """
        super().__init__()
        self.dist = dist
        self.name = str(barcode)
        self.add_feature("barcode", barcode)
        self.add_feature("cell_state", cell_state)
        self.add_feature("dead", dead)

