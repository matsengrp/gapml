import unittest
import tensorflow as tf

import numpy as np

from clt_likelihood_model import CLTLikelihoodModel
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from indel_sets import TargetTract
from allele_events import AlleleEvents
from target_status import TargetStatus, TargetDeactTract

class LikelihoodModelTestCase(unittest.TestCase):
    def setUp(self):
        self.num_targets = 10
        #topology = CellLineageTree(allele_events_list = [AlleleEvents(num_targets=self.num_targets)])
        #topology.add_feature("node_id", 0)
        bcode_metadata = BarcodeMetadata()
        self.sess = tf.InteractiveSession()
        self.mdl = CLTLikelihoodModel(
                None,
                bcode_metadata,
                self.sess,
                target_lams = 0.1 + np.arange(bcode_metadata.n_targets))

        tf.global_variables_initializer().run()
        self.target_lams = self.mdl.target_lams.eval()
        self.trim_long_probs = self.mdl.trim_long_probs.eval()

    def test_get_hazard(self):
        tt = TargetTract(2,2,2,2)
        hazard = self.mdl.get_hazard(tt)
        self.assertTrue(np.isclose(hazard,
            self.target_lams[2] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))

        tt = TargetTract(0,0,2,2)
        hazard = self.mdl.get_hazard(tt)
        self.assertTrue(np.isclose(hazard,
            self.target_lams[0] * self.target_lams[2] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))

    def _get_transition_target_status_hazard(
            self,
            target_stat_start,
            target_stat_end):
        hazard = self.sess.run(
            self.mdl.target_status_transition_hazards[
                self.mdl.target_status_transition_idxs[target_stat_start][target_stat_end]])
        return hazard

    def test_create_hazard(self):
        target_stat_start = TargetStatus(TargetDeactTract(1,9))
        target_stat_end = TargetStatus(TargetDeactTract(0,9))
        hazard = self._get_transition_target_status_hazard(
                target_stat_start,
                target_stat_end)
        print(hazard)
        print(self.target_lams[0] * (1 - self.trim_long_probs[0]))
        self.assertTrue(np.isclose(
            hazard,
            self.target_lams[0] * (1 - self.trim_long_probs[0])))

#    def test_create_hazard_away(self):
#        tts_list = [
#                (TargetTract(1,1,9,9),),
#                (TargetTract(0,1,7,8),),
#                (TargetTract(0,1,1,1), TargetTract(3,4,8,9)),
#                (TargetTract(0,1,1,1), TargetTract(4,4,8,9))]
#
#        hazard_away_node = self.mdl._create_hazard_away_nodes(tts_list)
#        hazard_aways = self.sess.run(hazard_away_node)
#
#        self.assertTrue(np.isclose(hazard_aways[0],
#            self.target_lams[0] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))
#
#        self.assertTrue(np.isclose(hazard_aways[1],
#            self.target_lams[9] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))
#
#        self.assertTrue(np.isclose(hazard_aways[2],
#            self.target_lams[2] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))
#
#        self.assertTrue(np.isclose(hazard_aways[3],
#            self.target_lams[2] * (1 - self.trim_long_probs[0])
#            + self.target_lams[2] * (1 - self.trim_long_probs[0]) * self.target_lams[3] * (1 - self.trim_long_probs[1])
#            + self.target_lams[3] * (1 - self.trim_long_probs[1])))
#
#    def test_list_target_tracts(self):
#        active_any_targs = [0,1]
#        any_tts = CLTLikelihoodModel.get_possible_target_tracts(active_any_targs)
#        self.assertEqual(any_tts, set([
#            TargetTract(0,0,0,0),
#            TargetTract(0,0,0,1),
#            TargetTract(0,0,1,1),
#            TargetTract(0,1,1,1),
#            TargetTract(1,1,1,1)]))
#
#        active_any_targs = [0,5]
#        any_tts = CLTLikelihoodModel.get_possible_target_tracts(active_any_targs)
#        self.assertEqual(any_tts, set([
#            TargetTract(0,0,0,0),
#            TargetTract(0,0,5,5),
#            TargetTract(5,5,5,5)]))
#
#    def test_create_transition_matrix(self):
#        key_list = [(), (TargetTract(1,1,1,1),), (TargetTract(1,1,1,2),), (TargetTract(1,1,1,1), TargetTract(2,2,2,2))]
#        mat_wrap = TransitionMatrixWrapper({
#            (): {
#                key_list[1]: TargetTract(1,1,1,1),
#                key_list[2]: TargetTract(1,1,1,2),
#                },
#            key_list[1]: {key_list[3]: TargetTract(2,2,2,2)},
#            key_list[2]: {},
#            key_list[3]: {}
#        }, key_list)
#        hazard_dict, _, _ = self.mdl._create_hazard_dict([mat_wrap])
#        hazard_away_dict, _ = self.mdl._create_hazard_away_dict([mat_wrap])
#        q_mat_node = self.mdl._create_transition_matrix(mat_wrap, hazard_dict, hazard_away_dict)
#        transition_matrix = self.sess.run(q_mat_node)
#
#        hazard_away_node = self.mdl._create_hazard_away_nodes(key_list)
#        hazard_aways = self.sess.run(hazard_away_node)
#
#        self.assertTrue(np.isclose(
#            transition_matrix[0][1],
#            self.target_lams[1] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))
#        self.assertTrue(np.isclose(
#            transition_matrix[0][2],
#            self.target_lams[1] * (1 - self.trim_long_probs[0]) * self.trim_long_probs[1]))
#        self.assertTrue(np.isclose(transition_matrix[0][3], 0))
#        self.assertTrue(np.isclose(
#            transition_matrix[0][4], hazard_aways[0] - self.target_lams[1] * (1 - self.trim_long_probs[0])))
#        self.assertTrue(np.isclose(
#            transition_matrix[0][0], -hazard_aways[0]))
#        self.assertTrue(np.isclose(
#            transition_matrix[1][0], 0))
#        self.assertTrue(np.isclose(
#            transition_matrix[1][1], -hazard_aways[1]))
#        self.assertTrue(np.isclose(
#            transition_matrix[1][2], 0))
#        self.assertTrue(np.isclose(
#            transition_matrix[1][3],
#            self.target_lams[2] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))
#        self.assertTrue(np.isclose(
#            transition_matrix[3][4], hazard_aways[3]))
#        # Unlikely row is all zeros
#        self.assertTrue(np.isclose(
#            np.max(np.abs(transition_matrix[4])), 0))
