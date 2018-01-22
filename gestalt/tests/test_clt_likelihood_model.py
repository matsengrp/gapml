import unittest

import numpy as np

from clt_likelihood_model import CLTLikelihoodModel
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from approximator import TransitionGraph, TransitionToNode
from indel_sets import *

class LikelihoodModelTestCase(unittest.TestCase):
    def setUp(self):
        topology = CellLineageTree()
        bcode_metadata = BarcodeMetadata()
        self.mdl = CLTLikelihoodModel(topology, bcode_metadata)

    def test_matching_singletons(self):
        anc_state = AncState(set([Wildcard(0,1), SingletonWC(40,3, 2,2,2,3)]))
        tts = (TargetTract(2,2,2,2),)
        matches = CLTLikelihoodModel.get_matching_singletons(anc_state, tts)
        self.assertEqual(len(matches), 0)

        anc_state = AncState(set([Wildcard(0,1), SingletonWC(40,3, 2,2,2,3)]))
        tts = (TargetTract(0,0,1,1), TargetTract(2,2,2,3),)
        matches = CLTLikelihoodModel.get_matching_singletons(anc_state, tts)
        self.assertEqual(matches, [Singleton(40,3, 2,2,2,3)])

    def test_get_hazard(self):
        self.mdl.target_lams = np.arange(self.mdl.num_targets)

        tt = TargetTract(2,2,2,2)
        hazard = self.mdl.get_hazard(tt)
        self.assertEqual(hazard,
            self.mdl.target_lams[2] * (1 - self.mdl.trim_long_probs[0]) * (1 - self.mdl.trim_long_probs[1]))

        tt = TargetTract(0,0,2,2)
        hazard = self.mdl.get_hazard(tt)
        self.assertEqual(hazard,
            self.mdl.target_lams[0] * self.mdl.target_lams[2] * (1 - self.mdl.trim_long_probs[0]) * (1 - self.mdl.trim_long_probs[1]))

    def test_get_hazard_away(self):
        self.mdl.target_lams = np.arange(self.mdl.num_targets)

        tts = (TargetTract(1,1,9,9),)
        hazard = self.mdl.get_hazard_away(tts)
        self.assertEqual(hazard,
            self.mdl.target_lams[0] * (1 - self.mdl.trim_long_probs[0]) * (1 - self.mdl.trim_long_probs[1]))

        tts = (TargetTract(0,1,7,8),)
        hazard = self.mdl.get_hazard_away(tts)
        self.assertEqual(hazard,
            self.mdl.target_lams[9] * (1 - self.mdl.trim_long_probs[0]) * (1 - self.mdl.trim_long_probs[1]))

        tts = (TargetTract(0,1,1,1), TargetTract(3,4,8,9))
        hazard = self.mdl.get_hazard_away(tts)
        self.assertEqual(hazard,
            self.mdl.target_lams[2] * (1 - self.mdl.trim_long_probs[0]) * (1 - self.mdl.trim_long_probs[1]))

        tts = (TargetTract(0,1,1,1), TargetTract(4,4,8,9))
        hazard = self.mdl.get_hazard_away(tts)
        self.assertEqual(hazard,
            self.mdl.target_lams[2] * (1 - self.mdl.trim_long_probs[0])
            + self.mdl.target_lams[2] * (1 - self.mdl.trim_long_probs[0]) * self.mdl.target_lams[3] * (1 - self.mdl.trim_long_probs[1])
            + self.mdl.target_lams[3] * (1 - self.mdl.trim_long_probs[1]))

    def test_partition(self):
        tts = (TargetTract(8,8,8,8),)
        anc_state = AncState([
            Wildcard(1,4),
            Wildcard(8,8)])
        parts = CLTLikelihoodModel.partition(tts, anc_state)
        self.assertEqual(parts[Wildcard(1,4)], ())
        self.assertEqual(parts[Wildcard(8,8)], (tts[0],))

        tts = (
            TargetTract(1,1,2,2),
            TargetTract(3,3,3,4),
            TargetTract(8,8,8,8))
        anc_state = AncState([
            Wildcard(1,4),
            Wildcard(8,9)])
        parts = CLTLikelihoodModel.partition(tts, anc_state)
        self.assertEqual(parts[Wildcard(1,4)], (tts[0], tts[1]))
        self.assertEqual(parts[Wildcard(8,9)], (tts[2],))

        tts = ()
        anc_state = AncState([
            Wildcard(1,4),
            Wildcard(8,9)])
        parts = CLTLikelihoodModel.partition(tts, anc_state)
        self.assertEqual(parts[Wildcard(1,4)], ())
        self.assertEqual(parts[Wildcard(8,9)], ())

    def test_list_target_tracts(self):
        active_any_targs = [0,1]
        any_tts = CLTLikelihoodModel.get_possible_target_tracts(active_any_targs)
        self.assertEqual(any_tts, set([
            TargetTract(0,0,0,0),
            TargetTract(0,0,0,1),
            TargetTract(0,0,1,1),
            TargetTract(0,1,1,1),
            TargetTract(1,1,1,1)]))

        active_any_targs = [0,5]
        any_tts = CLTLikelihoodModel.get_possible_target_tracts(active_any_targs)
        self.assertEqual(any_tts, set([
            TargetTract(0,0,0,0),
            TargetTract(0,0,5,5),
            TargetTract(5,5,5,5)]))

    def test_transition_dict_row(self):
        node1 = (TargetTract(3,3,3,3),)
        node2 = (TargetTract(2,2,4,4),)
        node3 = (TargetTract(1,1,5,5),)
        node4 = (TargetTract(0,0,9,9),)
        graph = TransitionGraph()
        graph.add_node(())
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        graph.add_node(node4)
        graph.add_edge((), TransitionToNode(node1[0], node1))
        graph.add_edge((), TransitionToNode(node2[0], node2))
        graph.add_edge((), TransitionToNode(node3[0], node3))
        graph.add_edge(node2, TransitionToNode(node3[0], node3))
        graph.add_edge(node3, TransitionToNode(node4[0], node4))
        tts_partition_info = {
            node4: {
                "start": (),
                "graph": graph
                }}
        indel_set_list = [node4]
        transition_dict = dict()
        self.mdl._add_transition_dict_row(tts_partition_info, indel_set_list, transition_dict)
        self.assertTrue(() in transition_dict[()])
        self.assertTrue(node1 in transition_dict[()])
        self.assertTrue(node2 in transition_dict[()])
        self.assertTrue(node3 in transition_dict[()])
        self.assertEqual(len(transition_dict[()].keys()), 5)
        self.assertTrue(node3 in transition_dict[node2])
        self.assertTrue(node2 in transition_dict[node2])
        self.assertTrue("unlikely" in transition_dict[node2])
        self.assertEqual(len(transition_dict[node2].keys()), 3)
        self.assertTrue(node4 in transition_dict[node3])
        self.assertEqual(len(transition_dict[node3].keys()), 3)
        self.assertEqual(transition_dict[node4]["unlikely"], 0)
        self.assertEqual(transition_dict[node4][node4], 0)
        self.assertEqual(len(transition_dict[node4].keys()), 2)
