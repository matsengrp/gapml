import unittest
import tensorflow as tf

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
        self.sess = tf.InteractiveSession()
        self.mdl = CLTLikelihoodModel(
                topology,
                bcode_metadata,
                self.sess,
                target_lams = np.arange(bcode_metadata.n_targets))

        tf.global_variables_initializer().run()
        self.target_lams = self.mdl.target_lams.eval()
        self.trim_long_probs = self.mdl.trim_long_probs.eval()

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
        tt = TargetTract(2,2,2,2)
        hazard = self.mdl.get_hazard(tt)
        self.assertTrue(np.isclose(hazard,
            self.target_lams[2] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))

        tt = TargetTract(0,0,2,2)
        hazard = self.mdl.get_hazard(tt)
        self.assertTrue(np.isclose(hazard,
            self.target_lams[0] * self.target_lams[2] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))

    def test_get_hazard_away(self):
        tts = (TargetTract(1,1,9,9),)
        hazard = self.mdl.get_hazard_away(tts)
        self.assertTrue(np.isclose(hazard,
            self.target_lams[0] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))

        tts = (TargetTract(0,1,7,8),)
        hazard = self.mdl.get_hazard_away(tts)
        self.assertTrue(np.isclose(hazard,
            self.target_lams[9] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))

        tts = (TargetTract(0,1,1,1), TargetTract(3,4,8,9))
        hazard = self.mdl.get_hazard_away(tts)
        self.assertTrue(np.isclose(hazard,
            self.target_lams[2] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))

        tts = (TargetTract(0,1,1,1), TargetTract(4,4,8,9))
        hazard = self.mdl.get_hazard_away(tts)
        self.assertTrue(np.isclose(hazard,
            self.target_lams[2] * (1 - self.trim_long_probs[0])
            + self.target_lams[2] * (1 - self.trim_long_probs[0]) * self.target_lams[3] * (1 - self.trim_long_probs[1])
            + self.target_lams[3] * (1 - self.trim_long_probs[1])))

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
