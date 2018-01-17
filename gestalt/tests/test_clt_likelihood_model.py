import unittest

import numpy as np

from clt_likelihood_model import CLTLikelihoodModel
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
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
