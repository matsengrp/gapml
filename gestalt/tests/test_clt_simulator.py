import unittest
import numpy as np

from cell_state import CellTypeTree, CellState
from cell_state_simulator import CellTypeSimulator
from cell_lineage_tree import CellLineageTree
from clt_simulator import CLTSimulator
from barcode_simulator import BarcodeSimulator
from barcode import Barcode

class CLTSimulatorTestCase(unittest.TestCase):
    def setUp(self):
        #Create cell type tree
        self.cell_type_tree = CellTypeTree(cell_type=None, rate=0)
        self.cell_type_tree.add_child(
            CellTypeTree(cell_type=0, rate=0.05))
        self.cell_type_tree.add_child(
            CellTypeTree(cell_type=1, rate=0.05))

        # Create barcode simulator
        self.bcode_simulator = BarcodeSimulator(
            target_lambdas=np.ones(10) * 0.1,
            repair_rates=[1, 2],
            indel_probability=0.1,
            left_del_mu=1,
            right_del_mu=1,
            insertion_mu=1)

        # Cell State simulator
        self.cell_state_simulator = CellTypeSimulator(
            self.cell_type_tree)

        # Create CLT simulator
        self.clt_simulator = CLTSimulator(
            birth_rate=1.0,
            death_rate=0.1,
            cell_state_simulator=self.cell_state_simulator,
            bcode_simulator=self.bcode_simulator)

    def test_basic(self):
        # Check that tree depth is what we requested
        simulation_time = 2
        tree = self.clt_simulator.simulate(
                Barcode(), CellState(categorical=self.cell_type_tree), simulation_time)
        farthest_node, farthest_dist = tree.get_farthest_node()
        self.assertTrue(np.isclose(farthest_dist, simulation_time))

    def test_process_birth(self):
        tree = self.clt_simulator.simulate(
                Barcode(), CellState(categorical=self.cell_type_tree), 0)
        leaf = tree.get_leaves()[0]
        new_barcode = self.bcode_simulator.simulate(leaf.barcode, time=1)
        self.clt_simulator._process_cell_birth(
            leaf,
            branch_length=1,
            branch_end_cell_state=leaf.cell_state,
            branch_end_barcode=new_barcode,
            remain_time=0.5)
        children = leaf.get_children()
        self.assertEqual(len(children), 1)
        for c in children:
            self.assertEqual(len(c.children), 2)
            self.assertEqual(c.dist, 1)
            self.assertEqual(c.dead, False)
            self.assertEqual(c.cell_state, leaf.cell_state)

    def test_process_end(self):
        tree = self.clt_simulator.simulate(
                Barcode(), CellState(categorical=self.cell_type_tree), 0)
        leaf = tree.get_leaves()[0]
        new_barcode = self.bcode_simulator.simulate(leaf.barcode, time=1)
        self.clt_simulator._process_observe_end(
            leaf,
            branch_length=1,
            branch_end_cell_state=leaf.cell_state,
            branch_end_barcode=new_barcode)
        children = leaf.get_children()
        desc_types = leaf.cell_state.categorical_state.get_children()
        self.assertEqual(len(children), 1)
        for c in children:
            self.assertEqual(c.dist, 1)
            self.assertEqual(c.dead, False)
            self.assertTrue(c.cell_state, leaf.cell_state)

    def test_process_death(self):
        tree = self.clt_simulator.simulate(
                Barcode(), CellState(categorical=self.cell_type_tree), 0)
        leaf = tree.get_leaves()[0]
        new_barcode = self.bcode_simulator.simulate(leaf.barcode, time=1)
        self.clt_simulator._process_cell_death(
            leaf,
            branch_length=1,
            branch_end_cell_state=leaf.cell_state,
            branch_end_barcode=new_barcode)
        children = leaf.get_children()
        desc_types = leaf.cell_state.categorical_state.get_children()
        self.assertEqual(len(children), 1)
        for c in children:
            self.assertEqual(c.dist, 1)
            self.assertEqual(c.dead, True)
            self.assertTrue(c.cell_state, leaf.cell_state)
