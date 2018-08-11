import numpy as np
import logging

from cell_lineage_tree import CellLineageTree
from clt_simulator import CLTSimulator

class CLTSimulatorSimplest(CLTSimulator):
    """
    Creates a cell lineage tree that is a single leaf:
    root -- leaf
    """
    def simulate(self,
            tree_seed: int,
            data_seed: int,
            tot_time: float,
            max_nodes: int = 10):
        np.random.seed(tree_seed)
        root_allele = self.allele_simulator.get_root()
        root_cell_state = self.cell_state_simulator.get_root()

        child_dist = tot_time

        tree = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=0)
        tree1 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=tot_time)
        tree.add_child(tree1)

        np.random.seed(data_seed)
        # Run the simulation to create the alleles along the tree topology
        self._simulate_alleles(tree)
        self._simulate_cell_states(tree)
        return tree


class CLTSimulatorSimpler(CLTSimulator):
    """
    Creates a cell lineage tree that is
    a simple tree:
    root -- child -- leaf1, leaf2, leaf3, leaf4
    """
    def simulate(self,
            tree_seed: int,
            data_seed: int,
            tot_time: float,
            max_nodes: int = 10):
        np.random.seed(tree_seed)
        root_allele = self.allele_simulator.get_root()
        root_cell_state = self.cell_state_simulator.get_root()

        child_dist = 0.5 * tot_time

        tree = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=0)
        tree1 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=child_dist)
        leaf1 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=tot_time - child_dist)
        leaf2 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=tot_time - child_dist)
        leaf3 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=tot_time - child_dist)
        leaf4 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=tot_time - child_dist)
        tree.add_child(tree1)
        tree1.add_child(leaf1)
        tree1.add_child(leaf2)
        tree1.add_child(leaf3)
        tree1.add_child(leaf4)
        logging.info("CHILD DIST %f tot time %f", child_dist, tot_time)
        print("CHILD DIST", child_dist, "tot time", tot_time)

        np.random.seed(data_seed)
        # Run the simulation to create the alleles along the tree topology
        self._simulate_alleles(tree)
        self._simulate_cell_states(tree)
        return tree

class CLTSimulatorSimple(CLTSimulator):
    """
    Creates a cell lineage tree that is
    a simple tree:
    root -- leaf1
        |
         -- child -- leaf2, leaf3
    """
    def simulate(self,
            tree_seed: int,
            data_seed: int,
            tot_time: float,
            max_nodes: int = 10):
        np.random.seed(tree_seed)
        root_allele = self.allele_simulator.get_root()
        root_cell_state = self.cell_state_simulator.get_root()

        child_dist = np.random.rand() * tot_time
        offset = child_dist/2

        tree = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=0)
        tree1 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=offset)
        leaf1 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=tot_time - offset)
        child = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=offset)
        leaf2 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=tot_time - child_dist)
        leaf3 = CellLineageTree(
            allele_list=root_allele,
            cell_state=root_cell_state,
            dist=tot_time - child_dist)
        tree.add_child(tree1)
        tree1.add_child(leaf1)
        tree1.add_child(child)
        child.add_child(leaf2)
        child.add_child(leaf3)

        np.random.seed(data_seed)
        # Run the simulation to create the alleles along the tree topology
        self._simulate_alleles(tree)
        self._simulate_cell_states(tree)
        return tree
