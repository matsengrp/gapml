from scipy.stats import expon, multinomial
import numpy as np

from cell_lineage_tree import CellLineageTree
from cell_state import CellTypeTree, CellState
from barcode import Barcode
from barcode_simulator import BarcodeSimulator

class CLTSimulator:
    def __init__(self, birth_rate: float, death_rate: float, cell_type_tree: CellTypeTree, bcode_simulator: BarcodeSimulator):
        self.birth_scale = 1.0/birth_rate
        self.death_scale = 1.0/death_rate
        self.cell_type_tree = cell_type_tree
        self.bcode_simulator = bcode_simulator

    def simulate(self, time: float):
        root_barcode = Barcode()
        cell_state = CellState(categorical=self.cell_type_tree)
        self.tree = CellLineageTree(root_barcode, cell_state, dist=0) 

        self._simulate_tree(self.tree, time)

    def _simulate_tree(self, tree: CellLineageTree, time: float):
        # Determine branch length and event at end of branch
        t_birth = expon.rvs(scale=self.birth_scale)
        t_death = expon.rvs(scale=self.death_scale)
        speciate_scale = tree.cell_state.categorical_state.scale
        t_speciate = expon.rvs(scale=speciate_scale) if speciate_scale is not None else np.inf
        race_winner = np.argmin([t_birth, t_death, t_speciate])
        branch_length = np.min([t_birth, t_death, t_speciate])
        remain_time = time - branch_length

        branch_end_barcode = self.bcode_simulator.simulate(tree.barcode, time=branch_length)

        if time < branch_length:
            # Time of event is past observation time
            child1 = CellLineageTree(branch_end_barcode, tree.cell_state, dist=branch_length)
            tree.add_child(child1)
        elif race_winner == 0:
            # Cell division
            child1 = CellLineageTree(branch_end_barcode, tree.cell_state, dist=branch_length)
            child2 = CellLineageTree(branch_end_barcode, tree.cell_state, dist=branch_length)
            tree.add_child(child1)
            tree.add_child(child2)
            self._simulate_tree(child1, remain_time)
            self._simulate_tree(child2, remain_time)
        elif race_winner == 1:
            # Cell death
            child1 = CellLineageTree(branch_end_barcode, tree.cell_state, dist=branch_length, dead=True)
            tree.add_child(child1)
        else:
            # Cell-type differentiation
            child_cell_types = tree.cell_state.categorical_state.children
            child_type_probs = [c.probability for c in child_cell_types]
            cell_type1, cell_type2 = np.random.choice(a=2, size=2, p=child_type_probs)
            child1 = CellLineageTree(branch_end_barcode, CellState(categorical=child_cell_types[cell_type1]), dist=branch_length)
            child2 = CellLineageTree(branch_end_barcode, CellState(categorical=child_cell_types[cell_type2]), dist=branch_length)
            tree.add_child(child1)
            tree.add_child(child2)
            self._simulate_tree(child1, remain_time)
            self._simulate_tree(child2, remain_time)

