import numpy as np

from cell_lineage_tree import CellLineageTree
from cell_state import CellTypeTree, CellState
from cell_state_simulator import CellStateSimulator
from allele import Allele
from allele_simulator import AlleleSimulator
from clt_simulator import CLTSimulator
from common import sigmoid

class CLTSimulatorOneLayer(CLTSimulator):
    """
    Class for simulating cell lineage trees.
    Used for testing things.
    """
    def simulate(self,
            tree_seed: int,
            data_seed: int,
            time: float,
            max_nodes: int = 10):
        """
        Generates a CLT based on the model

        @param time: amount of time to simulate the CLT
        @param max_layers: number of layers in this tree. at most 2
        """
        np.random.seed(tree_seed)
        np.random.seed(data_seed)

        root_allele = self.allele_simulator.get_root()
        root_cell_state = self.cell_state_simulator.get_root()
        tree = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=0)

        for i in range(max_nodes):
            child = CellLineageTree(
                allele_list=self.allele_simulator.get_root(),
                cell_state=self.cell_state_simulator.get_root(),
                dist=max(time + np.random.randn() * 0.3, 0.1))
            tree.add_child(child)

        print("treee", tree)
        self._simulate_on_branches(tree)
        self._label_leaves(tree)
        return tree

class CLTSimulatorTwoLayers(CLTSimulator):
    """
    Class for simulating cell lineage trees.
    Used for testing things.
    """
    def simulate(self,
            tree_seed: int,
            data_seed: int,
            time: float,
            max_nodes: int = 10):
        """
        Generates a CLT based on the model

        @param time: amount of time to simulate the CLT
        """
        np.random.seed(tree_seed)
        np.random.seed(data_seed)

        root_allele = self.allele_simulator.get_root()
        root_cell_state = self.cell_state_simulator.get_root()
        tree = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=0)

        for i in range(max_nodes):
            child = CellLineageTree(
                allele_list=self.allele_simulator.get_root(),
                cell_state=self.cell_state_simulator.get_root(),
                dist=max(time + np.random.randn() * 0.3, 0.1))
            tree.add_child(child)
            for i in range(2):
                child2 = CellLineageTree(
                    allele_list=self.allele_simulator.get_root(),
                    cell_state=self.cell_state_simulator.get_root(),
                    dist=max(time + np.random.randn() * 0.3, 0.1))
                child.add_child(child2)

        self._simulate_on_branches(tree)
        self._label_leaves(tree)
        return tree
