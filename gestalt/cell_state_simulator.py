from scipy.stats import expon
import numpy as np

from cell_lineage_tree import CellLineageTree
from cell_state import CellTypeTree, CellState

class CellStateSimulator:
    def simulate(self, cell_state: CellState, time: float):
        """
        @return CellState
        """
        raise NotImplementedError()

class CellTypeSimulator(CellStateSimulator):
    """
    Class for simulating speciation (or not)
    """
    def __init__(self, cell_type_tree: CellTypeTree):
        self.cell_type_tree = cell_type_tree
        # Create cell type dictionary
        self.cell_type_dict = dict()
        for node in cell_type_tree.traverse("preorder"):
            self.cell_type_dict[node.get_gen_name()] = node

    def _get_speciate_scale(self, cell_type: CellTypeTree):
        scale = cell_type.scale
        if scale is not None:
            return scale
        else:
            return np.inf

    def simulate(self, cell_state: CellState, time: float):
        remain_time = time
        curr_cell_type = cell_state.categorical_state
        while remain_time > 0:
            # Keep speciating until time runs out
            children = curr_cell_type.children
            if not children:
                # if no children states, we're at tip of cell type tree
                return CellState(categorical=curr_cell_type)

            t_speciates = [
                expon.rvs(scale=self._get_speciate_scale(c)) for c in children]
            race_winner = np.argmin(t_speciates)
            branch_length = np.min(t_speciates)
            remain_time = remain_time - branch_length
            if remain_time > 0:
                curr_cell_type = children[race_winner]
            else:
                # Ran out of time
                return CellState(categorical=curr_cell_type)


