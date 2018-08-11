import unittest
import tensorflow as tf

import numpy as np

from indel_sets import Singleton, TargetTract, Wildcard, SingletonWC
from clt_likelihood_model import CLTLikelihoodModel
from cell_lineage_tree import CellLineageTree
from barcode_metadata import BarcodeMetadata
from bounded_poisson import ZeroInflatedBoundedPoisson, PaddedBoundedPoisson
from target_status import TargetStatus, TargetDeactTract
from transition_wrapper_maker import TransitionWrapper
from anc_state import AncState
from optim_settings import KnownModelParams

class CLTTrimProbTestCase(unittest.TestCase):
    # TODO: add more tests to trimming code!

    def setUp(self):
        self.num_targets = 10
        self.bcode_metadata = BarcodeMetadata()
        self.known_params = KnownModelParams(tot_time=True)
        self.sess = tf.InteractiveSession()
        self.mdl = CLTLikelihoodModel(
                None,
                self.bcode_metadata,
                self.sess,
                known_params = self.known_params,
                target_lams = 0.1 + np.arange(self.bcode_metadata.n_targets))
        tf.global_variables_initializer().run()
        self.trim_zero_probs = self.mdl.trim_zero_probs.eval()
        # Suppose left and right are the same for now...
        self.trim_zero_prob = self.trim_zero_probs[0]
        self.trim_short_poiss = self.mdl.trim_short_poissons.eval()
        self.trim_long_poiss = self.mdl.trim_long_poissons.eval()

    def test_get_short_trim_probs(self):
        sg = Singleton(
                start_pos = 0,
                del_len = 20,
                min_deact_target = 0,
                min_target = 0,
                max_target = 0,
                max_deact_target = 0)
        left_trim, right_trim = sg.get_trim_lens(self.bcode_metadata)
        log_del_prob_node = self.mdl._create_log_del_probs([sg])
        log_del_prob = log_del_prob_node.eval()
        pois_left = ZeroInflatedBoundedPoisson(0, self.bcode_metadata.left_max_trim[0], self.trim_short_poiss[0])
        pois_right = ZeroInflatedBoundedPoisson(0, self.bcode_metadata.right_long_trim_min[0], self.trim_short_poiss[1])
        log_del = np.log(np.power(1 - self.trim_zero_prob, 2) * pois_left.pmf(left_trim) * pois_right.pmf(right_trim))
        self.assertTrue(np.isclose(
            log_del,
            log_del_prob[0]))

        sg = Singleton(
                start_pos = 262,
                del_len = 8,
                min_deact_target = 9,
                min_target = 9,
                max_target = 9,
                max_deact_target = 9)
        left_trim, right_trim = sg.get_trim_lens(self.bcode_metadata)
        log_del_prob_node = self.mdl._create_log_del_probs([sg])
        log_del_prob = log_del_prob_node.eval()
        pois_left = ZeroInflatedBoundedPoisson(0, self.bcode_metadata.left_long_trim_min[9], self.trim_short_poiss[0])
        pois_right = ZeroInflatedBoundedPoisson(0, self.bcode_metadata.right_long_trim_min[0] - 1, self.trim_short_poiss[1])
        log_del = np.log(
            (1 - self.trim_zero_prob) * pois_right.pmf(right_trim) * (
            self.trim_zero_prob + (1 - self.trim_zero_prob) * pois_left.pmf(left_trim)))
        self.assertTrue(np.isclose(
            log_del,
            log_del_prob[0]))

    def test_get_long_trim_probs(self):
        sg = Singleton(
                start_pos = 0,
                del_len = 42,
                min_deact_target = 0,
                min_target = 0,
                max_target = 0,
                max_deact_target = 1)
        left_trim, right_trim = sg.get_trim_lens(self.bcode_metadata)
        log_del_prob_node = self.mdl._create_log_del_probs([sg])
        log_del_prob = log_del_prob_node.eval()
        pois_left = ZeroInflatedBoundedPoisson(0, self.bcode_metadata.left_max_trim[0], self.trim_short_poiss[0])
        pois_right = PaddedBoundedPoisson(
                self.bcode_metadata.right_long_trim_min[0],
                self.bcode_metadata.right_max_trim[0],
                self.trim_long_poiss[1])
        log_del = np.log(
                (1 - self.trim_zero_prob) * pois_left.pmf(left_trim) * pois_right.pmf(right_trim))
        self.assertTrue(np.isclose(
            log_del,
            log_del_prob[0]))

        sg = Singleton(
                start_pos = 24,
                del_len = 45,
                min_deact_target = 0,
                min_target = 1,
                max_target = 1,
                max_deact_target = 2)
        left_trim, right_trim = sg.get_trim_lens(self.bcode_metadata)
        log_del_prob_node = self.mdl._create_log_del_probs([sg])
        log_del_prob = log_del_prob_node.eval()
        pois_left = PaddedBoundedPoisson(
                self.bcode_metadata.left_long_trim_min[sg.min_target],
                self.bcode_metadata.left_max_trim[sg.min_target],
                self.trim_short_poiss[0])
        pois_right = PaddedBoundedPoisson(
                self.bcode_metadata.right_long_trim_min[sg.max_target],
                self.bcode_metadata.right_max_trim[sg.max_target],
                self.trim_long_poiss[1])
        log_del = np.log(pois_left.pmf(left_trim) * pois_right.pmf(right_trim))
        self.assertTrue(np.isclose(
            log_del,
            log_del_prob[0]))

    def test_trim_prob_matrix(self):
        target_stat_start = TargetStatus()
        target_stat1 = TargetStatus(TargetDeactTract(0,1))
        target_stat2 = TargetStatus(TargetDeactTract(0,3))
        anc_state = AncState([Wildcard(0, 3)])
        t_wrap = TransitionWrapper(
                [target_stat_start, target_stat1, target_stat2],
                anc_state,
                is_leaf=True)
        trim_prob_mat_node = self.mdl._create_trim_prob_matrix(t_wrap)
        trim_prob_mat = trim_prob_mat_node.eval()
        self.assertTrue(np.all(trim_prob_mat == 1))

        target_stat_start = TargetStatus()
        target_stat_end = TargetStatus(TargetDeactTract(0,8))
        anc_state = AncState([SingletonWC(0, 236, 0, 0, 8, 8)])
        t_wrap = TransitionWrapper([target_stat_start, target_stat_end], anc_state, is_leaf=True)
        self.mdl._init_singleton_probs([Singleton(0, 236, 0, 0, 8, 8)])
        log_indel_prob = self.mdl.singleton_log_cond_prob[0].eval()
        trim_prob_mat_node = self.mdl._create_trim_prob_matrix(t_wrap)
        trim_prob_mat = trim_prob_mat_node.eval()
        self.assertTrue(np.isclose(
            np.log(trim_prob_mat[0, 1]),
            log_indel_prob
            ))
        self.assertEqual(trim_prob_mat[1, 0], 1)
