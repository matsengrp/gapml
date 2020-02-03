import unittest
import tensorflow as tf

import numpy as np
from scipy.stats import poisson

from indel_sets import Singleton
from clt_likelihood_model import CLTLikelihoodModel
from barcode_metadata import BarcodeMetadata
from bounded_distributions import ZeroInflatedBoundedPoisson, PaddedBoundedPoisson
from optim_settings import KnownModelParams


class CLTTrimProbTestCase(unittest.TestCase):
    def setUp(self):
        self.num_targets = 10
        self.bcode_meta = BarcodeMetadata()
        self.known_params = KnownModelParams(tot_time=True)
        self.sess = tf.InteractiveSession()
        #self.trim_short_params = np.array([10, 4, 8, 6])
        #self.trim_long_params = np.array([6, 5, 9, 3])
        self.trim_short_params = np.array([10, 4])
        self.trim_long_params = np.array([6, 5])
        self.trim_long_factor = np.array([0.5, 0.3, 0.3, 0.1])
        self.trim_zero_probs = np.array([0.25, 0.35, 0.08, 0.1])
        self.insert_params = np.array([1])
        self.insert_zero_prob = np.array([0.1])
        self.boost_softmax_weights = np.array([0.5,1,0.5])
        self.boost_probs = np.exp(self.boost_softmax_weights)/np.sum(np.exp(self.boost_softmax_weights))
        self.mdl = CLTLikelihoodModel(
                None,
                self.bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 0.1 + np.arange(self.bcode_meta.n_targets),
                boost_softmax_weights = self.boost_softmax_weights,
                trim_long_factor = self.trim_long_factor,
                trim_zero_probs = self.trim_zero_probs,
                trim_short_params = self.trim_short_params,
                trim_long_params = self.trim_long_params,
                insert_params = self.insert_params,
                insert_zero_prob = self.insert_zero_prob,
                use_poisson=True)
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
        left_del_prob = self.mdl._create_left_del_probs_short([sg], boost=0).eval()
        left_del_dist = ZeroInflatedBoundedPoisson(
                0,
                self.bcode_meta.left_long_trim_min[0] - 1,
                self.trim_short_params[0])
        left_del = (1 - self.trim_zero_probs[0]) * left_del_dist.pmf(left_trim)
        print(left_del, left_del_prob)
        self.assertTrue(np.isclose(left_del, left_del_prob))

        left_boost_del_prob = self.mdl._create_left_del_probs([sg], left_boost_len=1).eval()
        left_boost_del_dist = PaddedBoundedPoisson(
                1,
                self.bcode_meta.left_long_trim_min[0] - 1,
                self.trim_short_params[0])
        left_boost_del = left_boost_del_dist.pmf(left_trim)
        self.assertTrue(np.isclose(left_boost_del, left_boost_del_prob))

        right_del_prob = self.mdl._create_right_del_probs([sg], right_boost_len=0).eval()
        right_del_dist = ZeroInflatedBoundedPoisson(
                0,
                self.bcode_meta.right_long_trim_min[0] - 1,
                self.trim_short_params[2])
        right_del = (1 - self.trim_zero_probs[2]) * right_del_dist.pmf(right_trim)
        self.assertTrue(np.isclose(right_del, right_del_prob))

        right_boost_del_prob = self.mdl._create_right_del_probs([sg], right_boost_len=1).eval()
        right_boost_del_dist = PaddedBoundedPoisson(
                1,
                self.bcode_meta.right_long_trim_min[0] - 1,
                self.trim_short_params[2])
        right_boost_del = right_boost_del_dist.pmf(right_trim)
        self.assertTrue(np.isclose(right_boost_del, right_boost_del_prob))

        insert_prob = self.mdl._create_insert_probs([sg], insert_boost_len=0).eval()
        insert_dist = poisson(self.insert_params)
        insert = self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_dist.pmf(sg.insert_len)
        self.assertTrue(np.isclose(insert, insert_prob))

        log_indel_probs = self.mdl._create_log_indel_probs([sg]).eval()
        log_indel_self_calc = np.log(
                self.boost_probs[1] * insert_prob * right_del * left_boost_del
                + self.boost_probs[2] * insert_prob * right_boost_del * left_del)
        print(log_indel_probs, log_indel_self_calc)
        self.assertTrue(log_indel_probs, log_indel_self_calc)

    def test_get_short_trim_probs_rightmost_targ(self):
        sg = Singleton(
                start_pos = 262,
                del_len = 8,
                min_deact_target = 9,
                min_target = 9,
                max_target = 9,
                max_deact_target = 9,
                insert_str="ATAC")
        left_trim, right_trim = sg.get_trim_lens(self.bcode_meta)
        left_del_prob = self.mdl._create_left_del_probs([sg], left_boost_len=0).eval()
        left_del_dist = ZeroInflatedBoundedPoisson(
                0,
                self.bcode_meta.left_long_trim_min[9] - 1,
                self.trim_short_params[0])
        left_del = self.trim_zero_probs[0] + (1 - self.trim_zero_probs[0]) * left_del_dist.pmf(left_trim)
        self.assertTrue(np.isclose(left_del, left_del_prob[0]))

        right_del_prob = self.mdl._create_right_del_probs([sg], right_boost_len=0).eval()
        right_del_dist = ZeroInflatedBoundedPoisson(
                0,
                self.bcode_meta.right_long_trim_min[9] - 1,
                self.trim_short_params[2])
        right_del = (1 - self.trim_zero_probs[2]) * right_del_dist.pmf(right_trim)
        self.assertTrue(np.isclose(right_del, right_del_prob[0]))

        right_boost_del_dist = PaddedBoundedPoisson(
                1,
                self.bcode_meta.right_long_trim_min[9] - 1,
                self.trim_short_params[2])
        right_boost_del = right_boost_del_dist.pmf(right_trim)

        insert_prob = self.mdl._create_insert_probs([sg], insert_boost_len=0).eval()
        insert_dist = poisson(self.insert_params)
        insert = (1 - self.insert_zero_prob) * insert_dist.pmf(sg.insert_len) * 1./np.power(4,4)
        self.assertTrue(np.isclose(insert, insert_prob[0]))

        insert_boost_prob = self.mdl._create_insert_probs([sg], insert_boost_len=1).eval()
        insert_boost = insert_dist.pmf(sg.insert_len - 1) * 1./np.power(4,4)
        self.assertTrue(np.isclose(insert_boost, insert_boost_prob[0]))

        log_indel_probs = self.mdl._create_log_indel_probs([sg]).eval()
        log_indel_self_calc = np.log(
                self.boost_probs[2] * insert_prob * right_boost_del * left_del
                + self.boost_probs[0] * insert_boost_prob * right_del * left_del)
        print(log_indel_probs, log_indel_self_calc)
        self.assertTrue(np.isclose(log_indel_probs, log_indel_self_calc))

    def test_get_intertarg_trim_probs(self):
        sg = Singleton(
                start_pos = 228,
                del_len = 40,
                min_deact_target = 8,
                min_target = 8,
                max_target = 9,
                max_deact_target = 9,
                insert_str="ATAC")
        left_trim, right_trim = sg.get_trim_lens(self.bcode_meta)
        left_del_prob = self.mdl._create_left_del_probs_short([sg], boost=False).eval()
        left_del_dist = ZeroInflatedBoundedPoisson(
                0,
                self.bcode_meta.left_long_trim_min[8] - 1,
                self.trim_short_params[1])
        left_del = (1 - self.trim_zero_probs[1]) * left_del_dist.pmf(left_trim)
        print(left_del, left_del_prob)
        self.assertTrue(np.isclose(left_del, left_del_prob))

        right_del_prob = self.mdl._create_right_del_probs_any_long([sg]).eval()
        right_del_dist = ZeroInflatedBoundedPoisson(
                0,
                self.bcode_meta.right_long_trim_min[9] - 1,
                self.trim_short_params[3])
        right_del = (1 - self.trim_zero_probs[3]) * right_del_dist.pmf(right_trim)
        self.assertTrue(np.isclose(right_del, right_del_prob))

        insert_prob = self.mdl._create_insert_probs([sg]).eval()
        insert_dist = poisson(self.insert_params)
        insert = (1 - self.insert_zero_prob) * insert_dist.pmf(sg.insert_len) * 1./np.power(4,4)
        self.assertTrue(np.isclose(insert, insert_prob))

        log_indel_probs = self.mdl._create_log_indel_probs([sg]).eval()
        log_indel_self_calc = np.log(insert_prob * right_del * left_del)
        print(log_indel_probs, log_indel_self_calc)
        self.assertTrue(np.isclose(log_indel_probs, log_indel_self_calc))

    def test_get_long_trim_probs(self):
        sg = Singleton(
                start_pos = 22,
                del_len = 47,
                min_deact_target = 0,
                min_target = 1,
                max_target = 1,
                max_deact_target = 2,
                insert_str = "")
        left_trim, right_trim = sg.get_trim_lens(self.bcode_meta)
        print(left_trim, right_trim, sg.insert_len)
        print(sg.is_left_long, sg.is_right_long)

        left_del_prob = self.mdl._create_left_del_probs([sg]).eval()
        left_del_dist = PaddedBoundedPoisson(
                self.bcode_meta.left_long_trim_min[1],
                self.bcode_meta.left_max_trim[1],
                self.trim_long_params[0])
        left_del = left_del_dist.pmf(left_trim)
        print(left_del, left_del_prob)
        self.assertTrue(np.isclose(left_del, left_del_prob))

        right_del_prob = self.mdl._create_right_del_probs([sg]).eval()
        right_del_dist = PaddedBoundedPoisson(
                self.bcode_meta.right_long_trim_min[1],
                self.bcode_meta.right_max_trim[1],
                self.trim_long_params[2])
        right_del = right_del_dist.pmf(right_trim)
        print(right_del, right_del_prob)
        self.assertTrue(np.isclose(right_del, right_del_prob))

        insert_prob = self.mdl._create_insert_probs([sg], insert_boost_len=0).eval()
        insert_dist = poisson(self.insert_params)
        insert = self.insert_zero_prob + (1 - self.insert_zero_prob) * insert_dist.pmf(sg.insert_len)
        self.assertTrue(np.isclose(insert, insert_prob))

        log_indel_probs = self.mdl._create_log_indel_probs([sg]).eval()
        log_indel_self_calc = np.log(insert_prob * right_del * left_del)
        print(log_indel_probs, log_indel_self_calc)
        self.assertTrue(log_indel_probs, log_indel_self_calc)
