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
from optim_settings import KnownModelParams
from anc_state import AncState

class CLTTransitionProbTestCase(unittest.TestCase):
    def setUp(self):
        self.num_targets = 10
        self.double_cut_weight = 0.3
        bcode_metadata = BarcodeMetadata()
        self.known_params = KnownModelParams(tot_time=True)
        self.sess = tf.InteractiveSession()
        self.mdl = CLTLikelihoodModel(
                None,
                bcode_metadata,
                self.sess,
                known_params = self.known_params,
                target_lams = 0.1 + np.arange(bcode_metadata.n_targets),
                double_cut_weight = [self.double_cut_weight])

        tf.global_variables_initializer().run()
        self.target_lams = self.mdl.target_lams.eval()
        self.trim_long_factor = self.mdl.trim_long_factor.eval()
        self.trim_long_short_both = (1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1])

    def test_get_hazard_away(self):
        haz_away_node = self.mdl._create_hazard_away_target_statuses(
                [TargetStatus(TargetDeactTract(0,9))])
        haz_away = haz_away_node.eval()
        self.assertTrue(np.isclose(haz_away, 0))

        haz_away_node = self.mdl._create_hazard_away_target_statuses(
                [TargetStatus(TargetDeactTract(0,1), TargetDeactTract(3,9))])
        haz_away = haz_away_node.eval()
        self.assertTrue(np.isclose(haz_away,
            (1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1]) * self.target_lams[2]))

        haz_away_node = self.mdl._create_hazard_away_target_statuses(
                [TargetStatus(TargetDeactTract(0,1), TargetDeactTract(4,9))])
        haz_away = haz_away_node.eval()
        self.assertTrue(np.isclose(haz_away,
            (1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1]) * (
                np.sum(self.target_lams[2:4])
                + self.target_lams[2] * self.target_lams[3] * self.double_cut_weight)))

        haz_away_node = self.mdl._create_hazard_away_target_statuses(
            [TargetStatus(TargetDeactTract(0,1), TargetDeactTract(3,5), TargetDeactTract(7,9))])
        haz_away = haz_away_node.eval()
        self.assertTrue(np.isclose(haz_away,
            (1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1]) * (
                self.target_lams[2] + self.target_lams[6] +
                self.target_lams[2] * self.target_lams[6] * self.double_cut_weight)))

        haz_away_node = self.mdl._create_hazard_away_target_statuses(
            [TargetStatus(TargetDeactTract(0,1), TargetDeactTract(4,4), TargetDeactTract(7,9))])
        haz_away = haz_away_node.eval()
        my_haz_away = ((1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1]) * (
            (np.sum(self.target_lams[2:4]) + np.sum(self.target_lams[5:7])
            + np.sum(self.target_lams[2:4]) * np.sum(self.target_lams[5:7]) * self.double_cut_weight
            + self.target_lams[2] * self.target_lams[3] * self.double_cut_weight
            + self.target_lams[5] * self.target_lams[6] * self.double_cut_weight)))
        self.assertTrue(np.isclose(haz_away, my_haz_away))

    def test_get_hazard(self):
        tt_hazards = self.mdl.get_all_target_tract_hazards()
        tt = TargetTract(2,2,2,2)
        hazard = tt_hazards[self.mdl.target_tract_dict[tt]]
        self.assertTrue(np.isclose(hazard, self.target_lams[2]))

        tt = TargetTract(0,0,2,2)
        hazard = tt_hazards[self.mdl.target_tract_dict[tt]]
        self.assertTrue(np.isclose(hazard,
            self.double_cut_weight * self.target_lams[0] * self.target_lams[2] ))

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
                self.double_cut_weight * self.target_lams[0] * self.target_lams[1]
                + self.target_lams[0] * self.trim_long_factor[1]
                + self.target_lams[1] * self.trim_long_factor[0])
        self.assertTrue(np.isclose(hazard, q_mat[0, 1]))

        hazard = self.double_cut_weight * (
                self.target_lams[0] * self.target_lams[3]
                + self.target_lams[0] * self.target_lams[2] * self.trim_long_factor[1]
                + self.target_lams[1] * self.trim_long_factor[0] * self.target_lams[3]
                + self.target_lams[1] * self.trim_long_factor[0] * self.target_lams[2] * self.trim_long_factor[1])
        self.assertTrue(np.isclose(hazard, q_mat[0, 2]))

        hazard = (
                self.double_cut_weight * self.target_lams[2] * self.target_lams[3] * (1 + self.trim_long_factor[0])
                + self.target_lams[3] * self.trim_long_factor[0]
                + self.target_lams[2] * self.trim_long_factor[1] * (1 + self.trim_long_factor[0]))
        self.assertTrue(np.isclose(hazard, q_mat[1, 2]))

        self.assertTrue(np.isclose(0, q_mat[1, 0]))

        hazard = (
                 (1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1]) * self.double_cut_weight * (
                    0.5 * np.power(np.sum(self.target_lams[4:9]), 2)
                    - 0.5 * np.sum(np.power(self.target_lams[4:9], 2)))
                + (1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1]) * np.sum(self.target_lams[4:9])
                + (1 + self.trim_long_factor[0]) * self.target_lams[9]
                + (1 + self.trim_long_factor[0]) * self.double_cut_weight * np.sum(self.target_lams[4:9]) * self.target_lams[9])
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
             (1 + self.trim_long_factor[0]) * -self.target_lams[9],
             q_mat[1, 1]))

    def test_create_transition_matrix_away_two_targs(self):
        target_stat_start = TargetStatus()
        target_stat_end = TargetStatus(TargetDeactTract(0,7))
        anc_state = AncState([SingletonWC(0, 300, 0, 0, 7, 7)])
        transition_wrapper = TransitionWrapper([target_stat_start, target_stat_end], anc_state, is_leaf=True)
        q_mat_node = self.mdl._create_transition_matrix(transition_wrapper)
        q_mat = self.sess.run(q_mat_node)

        for i in range(q_mat.shape[0]):
            self.assertTrue(np.isclose(0, np.sum(q_mat[i,:])))

        self.assertTrue(np.isclose(
            -self.target_lams[9] * (1 + self.trim_long_factor[0])
            -self.target_lams[8] * (1 + self.trim_long_factor[0]) * (1 + self.trim_long_factor[1])
            -self.target_lams[8] * self.target_lams[9] * (1 + self.trim_long_factor[0]) * self.double_cut_weight,
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
            self.double_cut_weight * self.target_lams[0] * self.target_lams[9],
            q_mat[0, 1]))

        hazard_away = (
                self.trim_long_short_both * self.double_cut_weight * (
                    0.5 * np.power(np.sum(self.target_lams[1:9]), 2)
                    - 0.5 * np.sum(np.power(self.target_lams[1:9], 2)))
                + self.trim_long_short_both * np.sum(self.target_lams[1:9])
                + (1 + self.trim_long_factor[1]) * (
                    self.target_lams[0]
                    + self.double_cut_weight * self.target_lams[0] * np.sum(self.target_lams[1:9]))
                + (1 + self.trim_long_factor[0]) * (
                    self.target_lams[9]
                    + self.double_cut_weight * self.target_lams[9] * np.sum(self.target_lams[1:9]))
                + self.double_cut_weight * self.target_lams[0] * self.target_lams[9])
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
            self.double_cut_weight * self.target_lams[2] * self.target_lams[3],
            q_mat[0, 1]))
        self.assertTrue(np.isclose(self.target_lams[5], q_mat[1, 2]))
        self.assertEqual(q_mat[0,2], 0)
        self.assertEqual(q_mat[2,0], 0)
