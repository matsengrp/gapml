import unittest
import tensorflow as tf

import numpy as np

from indel_sets import SingletonWC
from clt_likelihood_model import CLTLikelihoodModel
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from indel_sets import TargetTract, Wildcard
from allele_events import AlleleEvents
from target_status import TargetStatus, TargetDeactTract
from transition_wrapper_maker import TransitionWrapper
from anc_state import AncState

class CLTTransitionProbTestCase(unittest.TestCase):
    def setUp(self):
        self.num_targets = 10
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
        tt_hazards = self.mdl.get_all_target_tract_hazards()
        tt = TargetTract(2,2,2,2)
        hazard = tt_hazards[self.mdl.target_tract_dict[tt]]
        self.assertTrue(np.isclose(hazard,
            self.target_lams[2] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))

        tt = TargetTract(0,0,2,2)
        hazard = tt_hazards[self.mdl.target_tract_dict[tt]]
        self.assertTrue(np.isclose(hazard,
            self.target_lams[0] * self.target_lams[2] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])))

    def test_create_transition_matrix(self):
        target_stat_start = TargetStatus()
        target_stat1 = TargetStatus(TargetDeactTract(0,1))
        target_stat2 = TargetStatus(TargetDeactTract(0,3))
        anc_state = AncState([Wildcard(0, 3)])
        transition_wrapper = TransitionWrapper(
                [target_stat_start, target_stat1, target_stat2],
                anc_state,
                is_leaf=True)
        q_mat_node = self.mdl._create_transition_matrix(transition_wrapper)
        q_mat = self.sess.run(q_mat_node)

        for i in range(q_mat.shape[0]):
            self.assertTrue(np.isclose(0, np.sum(q_mat[i,:])))

        hazard = (
                self.target_lams[0] * self.target_lams[1] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])
                + self.target_lams[0] * (1 - self.trim_long_probs[0]) * self.trim_long_probs[1]
                + self.target_lams[1] * self.trim_long_probs[0] * (1 - self.trim_long_probs[1]))
        self.assertTrue(np.isclose(hazard, q_mat[0, 1]))

        hazard = (
                self.target_lams[0] * self.target_lams[3] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1])
                + self.target_lams[0] * (1 - self.trim_long_probs[0]) * self.target_lams[2] * self.trim_long_probs[1]
                + self.target_lams[1] * self.trim_long_probs[0] * self.target_lams[3] * (1 - self.trim_long_probs[1])
                + self.target_lams[1] * self.trim_long_probs[0] * self.target_lams[2] * self.trim_long_probs[1])
        self.assertTrue(np.isclose(hazard, q_mat[0, 2]))

        hazard = (
                self.target_lams[2] * self.target_lams[3] * (1 - self.trim_long_probs[1])
                + self.target_lams[3] * self.trim_long_probs[0] * (1 - self.trim_long_probs[1])
                + self.target_lams[2] * self.trim_long_probs[1])
        self.assertTrue(np.isclose(hazard, q_mat[1, 2]))
        self.assertTrue(np.isclose(0, q_mat[1, 0]))

        hazard = (
                0.5 * np.power(np.sum(self.target_lams[4:9]), 2)
                - 0.5 * np.sum(np.power(self.target_lams[4:9], 2))
                + np.sum(self.target_lams[4:9])
                + self.target_lams[9] * (1 - self.trim_long_probs[1])
                + np.sum(self.target_lams[4:9]) * self.target_lams[9] * (1 - self.trim_long_probs[1]))
        self.assertTrue(np.isclose(-hazard, q_mat[2, 2]))
        self.assertTrue(np.isclose(0, q_mat[2, 0]))
        self.assertTrue(np.isclose(0, q_mat[2, 1]))

        self.assertTrue(np.all(q_mat[3,:] == np.zeros(4)))

    def test_create_transition_matrix_away_simple(self):
        target_stat_start = TargetStatus()
        target_stat_end = TargetStatus(TargetDeactTract(0,8))
        anc_state = AncState([SingletonWC(0, 300, 0, 0, 8, 8)])
        transition_wrapper = TransitionWrapper([target_stat_start, target_stat_end], anc_state, is_leaf=True)
        q_mat_node = self.mdl._create_transition_matrix(transition_wrapper)
        q_mat = self.sess.run(q_mat_node)

        for i in range(q_mat.shape[0]):
            self.assertTrue(np.isclose(0, np.sum(q_mat[i,:])))

        self.assertTrue(np.isclose(
            -self.target_lams[9] * (1 - self.trim_long_probs[1]),
            q_mat[1, 1]))

    def test_create_transition_matrix_with_singletonwc(self):
        target_stat_start = TargetStatus()
        target_stat_end = TargetStatus(TargetDeactTract(0,9))
        anc_state = AncState([SingletonWC(0, 300, 0, 0, 9, 9)])
        transition_wrapper = TransitionWrapper([target_stat_start, target_stat_end], anc_state, is_leaf=True)
        q_mat_node = self.mdl._create_transition_matrix(transition_wrapper)
        q_mat = self.sess.run(q_mat_node)

        for i in range(q_mat.shape[0]):
            self.assertTrue(np.isclose(0, np.sum(q_mat[i,:])))

        self.assertTrue(np.isclose(
            self.target_lams[0] * self.target_lams[9] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1]),
            q_mat[0, 1]))

        hazard_away = (0.5 * np.power(np.sum(self.target_lams[1:9]), 2)
                - 0.5 * np.sum(np.power(self.target_lams[1:9], 2))
                + np.sum(self.target_lams[1:9])
                + self.target_lams[0] * (1 - self.trim_long_probs[0])
                + self.target_lams[0] * (1 - self.trim_long_probs[0]) * np.sum(self.target_lams[1:9])
                + self.target_lams[9] * (1 - self.trim_long_probs[1])
                + self.target_lams[9] * (1 - self.trim_long_probs[1]) * np.sum(self.target_lams[1:9])
                + self.target_lams[0] * self.target_lams[9] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1]))
        self.assertTrue(np.isclose(-hazard_away, q_mat[0, 0]))

        self.assertEqual(q_mat[1,1], 0)

    def test_create_transition_matrix_mixed(self):
        target_stat_start = TargetStatus()
        target_stat1 = TargetStatus(TargetDeactTract(2,3))
        target_stat2 = TargetStatus(TargetDeactTract(2,3), TargetDeactTract(5,5))
        anc_state = AncState([
            SingletonWC(20, 40, 2, 2, 3, 3),
            Wildcard(5,5)])
        transition_wrapper = TransitionWrapper(
                [target_stat_start, target_stat1, target_stat2],
                anc_state,
                is_leaf=True)
        q_mat_node = self.mdl._create_transition_matrix(transition_wrapper)
        q_mat = self.sess.run(q_mat_node)

        for i in range(q_mat.shape[0]):
            self.assertTrue(np.isclose(0, np.sum(q_mat[i,:])))

        self.assertTrue(np.isclose(
            self.target_lams[2] * self.target_lams[3] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1]),
            q_mat[0, 1]))
        self.assertTrue(np.isclose(
            self.target_lams[5] * (1 - self.trim_long_probs[0]) * (1 - self.trim_long_probs[1]),
            q_mat[1, 2]))
        self.assertEqual(q_mat[0,2], 0)
        self.assertEqual(q_mat[2,0], 0)
