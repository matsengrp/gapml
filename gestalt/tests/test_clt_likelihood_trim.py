import unittest
import tensorflow as tf

import numpy as np
from scipy.stats import nbinom

from indel_sets import Singleton, Wildcard, SingletonWC
from clt_likelihood_model import CLTLikelihoodModel
from barcode_metadata import BarcodeMetadata
from bounded_distributions import ZeroInflatedBoundedNegativeBinomial, PaddedBoundedNegativeBinomial
from optim_settings import KnownModelParams
from common import sigmoid


class CLTTrimProbTestCase(unittest.TestCase):
    def setUp(self):
        self.num_targets = 10
        self.bcode_meta = BarcodeMetadata()
        self.known_params = KnownModelParams(tot_time=True)
        self.sess = tf.InteractiveSession()
        self.trim_short_nbinom_m = np.array([1, 2])
        self.trim_short_nbinom_logits = np.array([0, 0.5])
        self.trim_long_nbinom_m = np.array([4, 3])
        self.trim_long_nbinom_logits = np.array([-0.5, 1])
        self.trim_long_factor = np.array([0.2, 0.3])
        self.trim_zero_probs = np.array([0.25, 0.35])
        self.insert_nbinom_m = np.array([1])
        self.insert_nbinom_logit = np.array([0])
        self.insert_zero_prob = np.array([0.1])
        self.boost_softmax_weights = np.array([1,1,1])
        self.mdl = CLTLikelihoodModel(
                None,
                self.bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 0.1 + np.arange(self.bcode_meta.n_targets),
                boost_softmax_weights = self.boost_softmax_weights,
                trim_long_factor = self.trim_long_factor,
                trim_zero_probs = self.trim_zero_probs,
                trim_short_nbinom_m = self.trim_short_nbinom_m,
                trim_short_nbinom_logits = self.trim_short_nbinom_logits,
                trim_long_nbinom_m = self.trim_long_nbinom_m,
                trim_long_nbinom_logits = self.trim_long_nbinom_logits,
                insert_nbinom_m = self.insert_nbinom_m,
                insert_nbinom_logit = self.insert_nbinom_logit,
                insert_zero_prob = self.insert_zero_prob)
        tf.global_variables_initializer().run()
        self.mdl._create_trim_insert_distributions(1)

    def test_get_short_trim_probs_leftmost_targ(self):
        sg = Singleton(
                start_pos = 0,
                del_len = 20,
                min_deact_target = 0,
                min_target = 0,
                max_target = 0,
                max_deact_target = 0)
        left_trim, right_trim = sg.get_trim_lens(self.bcode_meta)
        print(left_trim, right_trim, sg.insert_len)
        left_del_prob = self.mdl._create_left_del_probs([sg], left_boost_len=0).eval()
        left_del_dist = ZeroInflatedBoundedNegativeBinomial(
                0,
                self.bcode_meta.left_max_trim[0],
                self.trim_short_nbinom_m[0],
                self.trim_short_nbinom_logits[0])
        left_del = (1 - self.trim_zero_probs[0]) * left_del_dist.pmf(left_trim)
        print(left_del, left_del_prob)
        self.assertTrue(np.isclose(left_del, left_del_prob[0]))

        left_boost_del_prob = self.mdl._create_left_del_probs([sg], left_boost_len=1).eval()
        left_boost_del_dist = ZeroInflatedBoundedNegativeBinomial(
                1,
                self.bcode_meta.left_max_trim[0],
                self.trim_short_nbinom_m[0],
                self.trim_short_nbinom_logits[0])
        left_boost_del = left_boost_del_dist.pmf(left_trim)
        print("left boost", left_boost_del, left_boost_del_prob)
        self.assertTrue(np.isclose(left_boost_del, left_boost_del_prob[0]))

        right_del_prob = self.mdl._create_right_del_probs([sg], right_boost_len=0).eval()
        right_del_dist = ZeroInflatedBoundedNegativeBinomial(
                0,
                self.bcode_meta.right_max_trim[0],
                self.trim_short_nbinom_m[1],
                self.trim_short_nbinom_logits[1])
        right_del = (1 - self.trim_zero_probs[1]) * right_del_dist.pmf(right_trim)
        print(right_del, right_del_prob)
        self.assertTrue(np.isclose(right_del, right_del_prob[0]))

        right_boost_del_prob = self.mdl._create_right_del_probs([sg], right_boost_len=1).eval()
        right_boost_del_dist = ZeroInflatedBoundedNegativeBinomial(
                1,
                self.bcode_meta.right_max_trim[0],
                self.trim_short_nbinom_m[1],
                self.trim_short_nbinom_logits[1])
        right_boost_del = right_boost_del_dist.pmf(right_trim)
        print("right boost", right_boost_del, right_boost_del_prob)
        self.assertTrue(np.isclose(right_boost_del, right_boost_del_prob[0], atol=1e-4))

        insert_prob = self.mdl._create_insert_probs([sg], insert_boost_len=0).eval()
        insert_dist = nbinom(self.insert_nbinom_m, 1 - sigmoid(self.insert_nbinom_logit))
        insert = self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_dist.pmf(sg.insert_len)
        print(insert, insert_prob)
        self.assertTrue(np.isclose(insert, insert_prob[0]))

        log_indel_probs = self.mdl._create_log_indel_probs([sg]).eval()
        log_indel_self_calc = 1./3 * (np.log(insert_prob) + np.log(right_del) + np.log(left_boost_del)) + 1./3 * (np.log(insert_prob) + np.log(right_boost_del) + np.log(left_del))
        print(log_indel_probs, log_indel_self_calc)
        self.assertTrue(log_indel_probs, log_indel_self_calc)

    @unittest.skip("Asdf")
    def test_get_short_trim_probs_rightmost_targ(self):
        sg = Singleton(
                start_pos = 262,
                del_len = 8,
                min_deact_target = 9,
                min_target = 9,
                max_target = 9,
                max_deact_target = 9)
        left_trim, right_trim = sg.get_trim_lens(self.bcode_meta)
        print(left_trim, right_trim, sg.insert_len)
        log_left_del_prob = self.mdl._create_left_del_probs([sg], left_boost_len=0).eval()
        left_del_dist = ZeroInflatedBoundedNegativeBinomial(
                0,
                self.bcode_meta.left_max_trim[0],
                self.trim_short_nbinom_m[0],
                self.trim_short_nbinom_logits[0])
        log_left_del = self.trim_zero_probs[0] + (1 - self.trim_zero_probs[0]) * left_del_dist.pmf(left_trim)
        print(log_left_del, log_left_del_prob)
        self.assertTrue(np.isclose(log_left_del, log_left_del_prob[0]))

        log_right_del_prob = self.mdl._create_right_del_probs([sg], right_boost_len=0).eval()
        right_del_dist = ZeroInflatedBoundedNegativeBinomial(
                0,
                self.bcode_meta.right_max_trim[0],
                self.trim_short_nbinom_m[1],
                self.trim_short_nbinom_logits[1])
        log_right_del = (1 - self.trim_zero_probs[1]) * right_del_dist.pmf(right_trim)
        print(log_right_del, log_right_del_prob)
        self.assertTrue(np.isclose(log_right_del, log_right_del_prob[0]))

        log_insert_prob = self.mdl._create_insert_probs([sg], insert_boost_len=0).eval()
        insert_dist = nbinom(self.insert_nbinom_m, 1 - sigmoid(self.insert_nbinom_logit))
        log_insert = self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_dist.pmf(sg.insert_len)
        print(log_insert, log_insert_prob)
        self.assertTrue(np.isclose(log_insert, log_insert_prob[0]))

#    def test_get_long_trim_probs(self):
#        sg = Singleton(
#                start_pos = 0,
#                del_len = 42,
#                min_deact_target = 0,
#                min_target = 0,
#                max_target = 0,
#                max_deact_target = 1)
#        left_trim, right_trim = sg.get_trim_lens(self.bcode_meta)
#        log_del_prob_node = self.mdl._create_log_del_probs([sg])
#        log_del_prob = log_del_prob_node.eval()
#        pois_left = ZeroInflatedBoundedPoisson(0, self.bcode_meta.left_max_trim[0], self.trim_short_poiss[0])
#        pois_right = PaddedBoundedPoisson(
#                self.bcode_meta.right_long_trim_min[0],
#                self.bcode_meta.right_max_trim[0],
#                self.trim_long_poiss[1])
#        log_del = np.log(
#                (1 - self.trim_zero_prob) * pois_left.pmf(left_trim) * pois_right.pmf(right_trim))
#        self.assertTrue(np.isclose(
#            log_del,
#            log_del_prob[0]))
#
#        sg = Singleton(
#                start_pos = 24,
#                del_len = 45,
#                min_deact_target = 0,
#                min_target = 1,
#                max_target = 1,
#                max_deact_target = 2)
#        left_trim, right_trim = sg.get_trim_lens(self.bcode_meta)
#        log_del_prob_node = self.mdl._create_log_del_probs([sg])
#        log_del_prob = log_del_prob_node.eval()
#        pois_left = PaddedBoundedPoisson(
#                self.bcode_meta.left_long_trim_min[sg.min_target],
#                self.bcode_meta.left_max_trim[sg.min_target],
#                self.trim_short_poiss[0])
#        pois_right = PaddedBoundedPoisson(
#                self.bcode_meta.right_long_trim_min[sg.max_target],
#                self.bcode_meta.right_max_trim[sg.max_target],
#                self.trim_long_poiss[1])
#        log_del = np.log(pois_left.pmf(left_trim) * pois_right.pmf(right_trim))
#        self.assertTrue(np.isclose(
#            log_del,
#            log_del_prob[0]))
#
#    def test_trim_prob_matrix(self):
#        target_stat_start = TargetStatus()
#        target_stat1 = TargetStatus(TargetDeactTract(0,1))
#        target_stat2 = TargetStatus(TargetDeactTract(0,3))
#        anc_state = AncState([Wildcard(0, 3)])
#        t_wrap = TransitionWrapper(
#                [target_stat_start, target_stat1, target_stat2],
#                anc_state,
#                is_leaf=True)
#        trim_prob_mat_node = self.mdl._create_trim_prob_matrix(t_wrap)
#        trim_prob_mat = trim_prob_mat_node.eval()
#        self.assertTrue(np.all(trim_prob_mat == 1))
#
#        target_stat_start = TargetStatus()
#        target_stat_end = TargetStatus(TargetDeactTract(0,8))
#        anc_state = AncState([SingletonWC(0, 236, 0, 0, 8, 8)])
#        t_wrap = TransitionWrapper([target_stat_start, target_stat_end], anc_state, is_leaf=True)
#        self.mdl._init_singleton_probs([Singleton(0, 236, 0, 0, 8, 8)])
#        log_indel_prob = self.mdl.singleton_log_cond_prob[0].eval()
#        trim_prob_mat_node = self.mdl._create_trim_prob_matrix(t_wrap)
#        trim_prob_mat = trim_prob_mat_node.eval()
#        self.assertTrue(np.isclose(
#            np.log(trim_prob_mat[0, 1]),
#            log_indel_prob
#            ))
#        self.assertEqual(trim_prob_mat[1, 0], 1)
