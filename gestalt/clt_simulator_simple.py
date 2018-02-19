import copy
from scipy.stats import expon
import numpy as np

from cell_lineage_tree import CellLineageTree
from cell_state import CellTypeTree, CellState
from cell_state_simulator import CellStateSimulator
from allele import Allele
from allele_events import AlleleEvents
from allele_simulator import AlleleSimulator

from common import sigmoid

class CLTSimulatorSimple:
    """
    Class for simulating cell lineage trees.
    Used for testing things.
    """
    def __init__(self,
        cell_state_simulator: CellStateSimulator,
        allele_simulator: AlleleSimulator):
        """
        @param cell_type_tree: the tree that specifies how cells differentiate
        @param allele_simulator: a simulator for how alleles get modified
        """
        self.cell_state_simulator = cell_state_simulator
        self.allele_simulator = allele_simulator

    def simulate(self, root_allele: Allele, root_cell_state: CellState, time: float, max_nodes: int = 10):
        """
        Generates a CLT based on the model

        @param time: amount of time to simulate the CLT
        """
        tree = CellLineageTree(
            allele=root_allele,
            cell_state=root_cell_state,
            dist=0)

        for i in range(max_nodes):
            child = self._simulate_branch(tree, time + np.random.rand() * 0.5)
            tree.add_child(child)
            for i in range(2):
                child2 = self._simulate_branch(child, time/2 + np.random.rand() * 0.5)
                child.add_child(child2)

        # Need to label the leaves (alive cells only) so that we
        # can later prune the tree. Leaf names must be unique!
        for idx, leaf in enumerate(tree, 1):
            if not leaf.dead:
                leaf.name = "leaf%d" % idx
        return tree

    def _simulate_branch(self, tree: CellLineageTree, time: float):
        """
        @param time: the max amount of time to simulate from this node
        """
        branch_end_cell_state = tree.cell_state

        branch_end_allele = self.allele_simulator.simulate(
            tree.allele,
            time=time)

        child = CellLineageTree(
            allele=branch_end_allele,
            cell_state=branch_end_cell_state,
            dist=time)
        return child
