from typing import List

from numpy import ndarray
from enum import Enum
from ete3 import TreeNode


class CellType(Enum):
    BRAIN = 1
    BLOOD = 2
    EYE = 3


class CellTypeTree(TreeNode):
    def __init__(self, cell_type: CellType=None, rate: float=0, probability: float=1.0):
        """
        @param cell_type: the cell type of this node is the union of labeled cell types of the
                            descendant nodes (including this node)
        @param rate: the cell-type differentiation rate
        @param probability: the probability of going from parent to this node
        """
        super().__init__()
        self.add_feature("cell_type", cell_type)
        if cell_type is not None:
            self.name = cell_type
        self.add_feature("rate", rate)
        self.add_feature("scale", 1.0/rate if rate > 0 else None)
        self.add_feature("probability", probability)

class CellState:
    def __init__(self, categorical: CellTypeTree=None, cts: ndarray=None):
        """
        @param categorical_state: cell state variable that is categorical
        @param cts_state: cell state variable that is continuous, e.g. [0.1, 0.3, ...]
        """
        self.categorical_state = categorical
        self.cts_state = cts

