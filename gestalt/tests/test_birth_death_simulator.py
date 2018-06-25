import unittest
import numpy as np

from clt_simulator import BirthDeathTreeSimulator
from allele import AlleleList
from barcode_metadata import BarcodeMetadata

class BirthDeathSimulatorTestCase(unittest.TestCase):
    def setUp(self):
        self.birth_death_simulator = BirthDeathTreeSimulator(
            birth_rate=0.5,
            death_rate=0.1)

    def test_basic(self):
        # Check that tree depth is what we requested
        simulation_time = 2
        tree = self.birth_death_simulator.simulate(
                AlleleList([], BarcodeMetadata()),
                simulation_time,
                max_nodes=300)
        farthest_node, farthest_dist = tree.get_farthest_node()
        self.assertTrue(np.isclose(farthest_dist, simulation_time))

    def test_process_birth(self):
        tree = self.birth_death_simulator.simulate(
                AlleleList([], BarcodeMetadata()),
                0)
        leaf = tree.get_leaves()[0]
        self.birth_death_simulator._process_cell_birth(
            leaf,
            branch_length=1,
            branch_end_allele_list=AlleleList([], BarcodeMetadata()),
            remain_time=0.5)
        children = leaf.get_children()
        self.assertEqual(len(children), 1)
        for c in children:
            self.assertEqual(len(c.children), 2)
            self.assertEqual(c.dist, 1)
            self.assertEqual(c.dead, False)

    def test_process_end(self):
        tree = self.birth_death_simulator.simulate(
                AlleleList([], BarcodeMetadata()),
                0)
        leaf = tree.get_leaves()[0]
        self.birth_death_simulator._process_observe_end(
            leaf,
            branch_length=1,
            branch_end_allele_list=AlleleList([], BarcodeMetadata()))
        children = leaf.get_children()
        self.assertEqual(len(children), 1)
        for c in children:
            self.assertEqual(c.dist, 1)
            self.assertEqual(c.dead, False)

    def test_process_death(self):
        tree = self.birth_death_simulator.simulate(
                AlleleList([], BarcodeMetadata()),
                0)
        leaf = tree.get_leaves()[0]
        self.birth_death_simulator._process_cell_death(
            leaf,
            branch_length=1,
            branch_end_allele_list=AlleleList([], BarcodeMetadata()))
        children = leaf.get_children()
        self.assertEqual(len(children), 1)
        for c in children:
            self.assertEqual(c.dist, 1)
            self.assertEqual(c.dead, True)
