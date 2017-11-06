from typing import List

from numpy import ndarray
from enum import IntEnum
from ete3 import TreeNode

from constants import COLORS

class CellTypeTree(TreeNode):
    """
    Stores a cell-type tree with parameters for generating cell lineage trees
    """

    def __init__(self,
                 cell_type: int = None,
                 rate: float = 0,
                 probability: float = 1.0):
        """
        @param cell_type: the cell type of this node is the union of labeled cell types of the
                            descendant nodes (including this node)
        @param rate: the cell-type differentiation rate
        @param probability: the probability of going from parent to this node
        """
        super().__init__()
        self.add_feature("cell_type", cell_type)
        if cell_type is not None:
            self.name = str(cell_type)
        self.add_feature("rate", rate)
        if self.rate is not None:
            self.add_feature("scale", 1.0 / rate if rate > 0 else None)
        self.add_feature("probability", probability)


class CellState:
    """
    A description of the cell state
    """

    def __init__(self, categorical: CellTypeTree = None, cts: ndarray = None):
        """
        @param categorical_state: cell state variable that is categorical, indicates which cell-type in cell-type tree
        @param cts_state: cell state variable that is continuous, e.g. gene expression data like [0.1, 0.3, ...]
        """
        self.categorical_state = categorical
        self.cts_state = cts

    def __str__(self):
        if self.cts_state is None:
            return ",".join([leaf.name for leaf in self.categorical_state])
        else:
            return str(cts)
