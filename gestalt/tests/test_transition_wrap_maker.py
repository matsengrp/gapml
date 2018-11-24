import unittest

from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from indel_sets import TargetTract
from allele_events import AlleleEvents, Event
from target_status import TargetStatus, TargetDeactTract
from transition_wrapper_maker import TransitionWrapperMaker

class TransitionWrapperMakerTestCase(unittest.TestCase):
    def setUp(self):
        self.num_targets = 4

    def _create_bcode(self, num_barcodes):
        bcode_orig = BarcodeMetadata.create_fake_barcode_str(self.num_targets)
        bcode_meta = BarcodeMetadata(unedited_barcode = bcode_orig, num_barcodes = num_barcodes)
        return bcode_meta

    def test_transition_nothing_happened(self):
        num_barcodes = 1
        bcode_meta = self._create_bcode(num_barcodes)

        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        child1 = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=10)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[0]), num_barcodes)
        self.assertEqual(transition_wrap_dict[0][0].states, [TargetStatus()])
        self.assertEqual(transition_wrap_dict[1][0].states, [TargetStatus()])

    def test_transition_one_thing_happened(self):
        num_barcodes = 1
        bcode_meta = self._create_bcode(num_barcodes)

        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        child1 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,10,0,0,"")], num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        # allow many extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=10)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[0]), num_barcodes)
        self.assertEqual(transition_wrap_dict[0][0].states, [TargetStatus()])
        self.assertEqual(transition_wrap_dict[1][0].states, [TargetStatus(), TargetStatus(TargetDeactTract(0,0))])

        # what happens when i dont allow any extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=0)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[0]), num_barcodes)
        self.assertEqual(transition_wrap_dict[0][0].states, [TargetStatus()])
        self.assertEqual(transition_wrap_dict[1][0].states, [TargetStatus(), TargetStatus(TargetDeactTract(0,0))])

    def test_transition_two_things_happened(self):
        num_barcodes = 1
        bcode_meta = self._create_bcode(num_barcodes)

        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        child1 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,10,0,0,""), Event(40,10,1,1,"a")], num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        # allow many extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=10)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[0]), num_barcodes)
        self.assertEqual(transition_wrap_dict[0][0].states, [TargetStatus()])
        self.assertEqual(len(transition_wrap_dict[1][0].states), 4)
        self.assertTrue(TargetStatus() in transition_wrap_dict[1][0].states)
        self.assertTrue(TargetStatus(TargetDeactTract(0,0)) in transition_wrap_dict[1][0].states)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in transition_wrap_dict[1][0].states)
        self.assertTrue(TargetStatus(TargetDeactTract(0,1)) in transition_wrap_dict[1][0].states)

        # what happens when i dont allow any extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=0)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[0]), num_barcodes)
        self.assertEqual(transition_wrap_dict[0][0].states, [TargetStatus()])
        self.assertEqual(len(transition_wrap_dict[1][0].states), 4)
        self.assertTrue(TargetStatus() in transition_wrap_dict[1][0].states)
        self.assertTrue(TargetStatus(TargetDeactTract(0,0)) in transition_wrap_dict[1][0].states)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in transition_wrap_dict[1][0].states)
        self.assertTrue(TargetStatus(TargetDeactTract(0,1)) in transition_wrap_dict[1][0].states)

    def test_transition_long_intertarget(self):
        num_barcodes = 1
        bcode_meta = self._create_bcode(num_barcodes)

        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)
        child1 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,100,0,2,"")], num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        # allow many extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=10)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].target_tract_tuples), 3)
        self.assertEqual(len(transition_wrap_dict[1][0].states), 3)

        # allow one extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=1)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].states), 3)
        self.assertEqual(len(transition_wrap_dict[1][0].target_tract_tuples), 3)

        # allow no extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=0)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].states), 2)

    def test_transition_super_long_intertarget(self):
        num_barcodes = 1
        bcode_meta = self._create_bcode(num_barcodes)

        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)
        child1 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,130,0,3,"")], num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        # allow many extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=2)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].states), 5)

        # allow one extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=1)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].states), 5)
        # hidden possible target tract tuples are: focal target 1 (short), focal target 2 (short), cut 1 spill over to 2.
        # cut 2 spill over to 1, inter-target 1 + 2
        self.assertEqual(len(transition_wrap_dict[1][0].target_tract_tuples), 7)

        # allow no extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=0)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].states), 2)

    def test_transition_tree_no_extra(self):
        """
        Test transition matrix on tree with no extra steps
        """
        num_barcodes = 1
        bcode_meta = self._create_bcode(num_barcodes)

        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        child1 = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        child2 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,10,0,0,"")], num_targets=self.num_targets)])
        child1.add_child(child2)
        child2.add_feature("node_id", 2)

        child3 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,10,0,0,""), Event(40,10,1,1,"")], num_targets=self.num_targets)])
        child1.add_child(child3)
        child3.add_feature("node_id", 3)

        # allow no extra steps
        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=0)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].states), 2)
        self.assertTrue(TargetStatus(TargetDeactTract(0,0)) in transition_wrap_dict[1][0].states)
        self.assertEqual(len(transition_wrap_dict[1][0].target_tract_tuples), 2)

        self.assertEqual(len(transition_wrap_dict[2][0].states), 1)

        self.assertEqual(len(transition_wrap_dict[3][0].states), 2)
        self.assertTrue(TargetStatus(TargetDeactTract(0,1)) in transition_wrap_dict[3][0].states)
        self.assertTrue(TargetStatus(TargetDeactTract(0,0)) in transition_wrap_dict[3][0].states)
        self.assertEqual(len(transition_wrap_dict[3][0].target_tract_tuples), 2)

    def test_transition_tree_allow_extra(self):
        """
        Test transition matrix on tree with many extra steps
        """
        num_barcodes = 1
        bcode_meta = self._create_bcode(num_barcodes)

        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        child1 = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        child2 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,10,0,0,"")], num_targets=self.num_targets)])
        child1.add_child(child2)
        child2.add_feature("node_id", 2)

        child3 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,10,0,0,""), Event(40,10,1,1,"")], num_targets=self.num_targets)])
        child1.add_child(child3)
        child3.add_feature("node_id", 3)

        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=10)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].states), 2)
        self.assertTrue(TargetStatus(TargetDeactTract(0,0)) in transition_wrap_dict[1][0].states)
        self.assertEqual(len(transition_wrap_dict[1][0].target_tract_tuples), 2)

        self.assertEqual(len(transition_wrap_dict[2][0].states), 2)
        self.assertTrue(TargetStatus(TargetDeactTract(0,0)) in transition_wrap_dict[2][0].states)

        self.assertEqual(len(transition_wrap_dict[3][0].states), 4)
        self.assertTrue(TargetStatus(TargetDeactTract(0,1)) in transition_wrap_dict[3][0].states)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in transition_wrap_dict[3][0].states)
        self.assertTrue(TargetStatus(TargetDeactTract(0,0)) in transition_wrap_dict[3][0].states)
        self.assertEqual(len(transition_wrap_dict[3][0].target_tract_tuples), 4)

    def test_transition_tree_allow_extra_intertarg(self):
        """
        Test transition matrix on tree with many extra steps
        """
        num_barcodes = 1
        bcode_meta = self._create_bcode(num_barcodes)

        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        child1 = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        child2 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,50,0,2,"a")], num_targets=self.num_targets)])
        child1.add_child(child2)
        child2.add_feature("node_id", 2)

        child3 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(5,50,0,2,"aa")], num_targets=self.num_targets)])
        child1.add_child(child3)
        child3.add_feature("node_id", 3)

        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=10)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].states), 2)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in transition_wrap_dict[1][0].states)
        self.assertEqual(len(transition_wrap_dict[1][0].target_tract_tuples), 2)

        self.assertEqual(len(transition_wrap_dict[2][0].states), 3)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in transition_wrap_dict[2][0].states)

        self.assertEqual(len(transition_wrap_dict[3][0].states), 3)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in transition_wrap_dict[3][0].states)
        self.assertEqual(len(transition_wrap_dict[3][0].target_tract_tuples), 3)

    def test_transition_tree_no_extra_intertarg(self):
        """
        Test transition matrix on tree with many extra steps
        """
        num_barcodes = 1
        bcode_meta = self._create_bcode(num_barcodes)

        topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        topology.add_feature("node_id", 0)

        child1 = CellLineageTree(allele_events_list=[AlleleEvents(num_targets=self.num_targets)])
        topology.add_child(child1)
        child1.add_feature("node_id", 1)

        child2 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(10,50,0,2,"a")], num_targets=self.num_targets)])
        child1.add_child(child2)
        child2.add_feature("node_id", 2)

        child3 = CellLineageTree(allele_events_list=[
            AlleleEvents([Event(40,10,1,1,"")], num_targets=self.num_targets)])
        child1.add_child(child3)
        child3.add_feature("node_id", 3)

        trans_wrap_maker = TransitionWrapperMaker(topology, bcode_meta, max_extra_steps=0)
        transition_wrap_dict = trans_wrap_maker.create_transition_wrappers()
        self.assertEqual(len(transition_wrap_dict[1][0].states), 2)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in transition_wrap_dict[1][0].states)
        self.assertEqual(len(transition_wrap_dict[1][0].target_tract_tuples), 2)

        self.assertEqual(len(transition_wrap_dict[2][0].states), 3)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in transition_wrap_dict[2][0].states)

        self.assertEqual(len(transition_wrap_dict[3][0].states), 2)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in transition_wrap_dict[3][0].states)
        self.assertEqual(len(transition_wrap_dict[3][0].target_tract_tuples), 2)
