import unittest
import numpy as np
import tensorflow as tf

from target_status import TargetStatus, TargetDeactTract
from allele_simulator_simult import AlleleSimulatorSimultaneous
from barcode_metadata import BarcodeMetadata
from clt_likelihood_model import CLTLikelihoodModel
from optim_settings import KnownModelParams
from cell_lineage_tree import CellLineageTree
from indel_sets import TargetTract
from common import sigmoid
from bounded_distributions import ZeroInflatedBoundedNegativeBinomial, PaddedBoundedNegativeBinomial


class AlleleSimulatorTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.bcode_meta = BarcodeMetadata()
        self.sess = tf.InteractiveSession()
        self.known_params = KnownModelParams(tot_time=True)

    def _create_simulator(self, mdl):
        tf.global_variables_initializer().run()

        self.allele_sim = AlleleSimulatorSimultaneous(mdl)
        self.allele_list_root = self.allele_sim.get_root()
        self.allele = self.allele_list_root.alleles[0]
        self.topology = CellLineageTree(self.allele_list_root)
        self.node = CellLineageTree(self.allele_list_root)
        self.topology.add_child(self.node)
        self.topology.label_dist_to_roots()

    def test_race_process_no_events(self):
        mdl = CLTLikelihoodModel(
                None,
                self.bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 1 + np.arange(self.bcode_meta.n_targets))
        self._create_simulator(mdl)

        # Nothing to repair/cut
        for i in range(self.allele.bcode_meta.n_targets):
            self.allele.indel(i, i, 0, 0, "att")
        race_winner, event_time = self.allele_sim._race_target_tracts(
            self.allele,
            scale_hazard = lambda x: 1)
        self.assertEqual(race_winner, None)
        self.assertEqual(event_time, None)

    def test_simulate(self):
        mdl = CLTLikelihoodModel(
                None,
                self.bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 1 + np.arange(self.bcode_meta.n_targets))
        self._create_simulator(mdl)
        old_allele_str = str(self.allele)

        # Simulate for zero time. Nothing should happen
        self.node.dist = 0
        new_allele = self.allele_sim.simulate(self.allele, self.node)
        self.assertEqual(str(new_allele), old_allele_str)

        # Simulate for long time
        # (This is an assumption this allele simulator makes!)
        self.node.dist = 10
        new_allele = self.allele_sim.simulate(self.allele, self.node)
        # Make sure the old allele is not affected
        self.assertEqual(old_allele_str, str(self.allele))
        # Allele should be contiguous
        self.assertEqual(new_allele.get_target_status(), TargetStatus(TargetDeactTract(0,9)))

    def test_neg_beta_insert_focal(self):
        insert_nbinom_m = np.array([1])
        insert_nbinom_logit = np.array([0])
        insert_nbinom_prob = sigmoid(insert_nbinom_logit)
        insert_zero_prob = np.array([0.1])
        boost_softmax_weights = np.array([1,1,1])
        mdl = CLTLikelihoodModel(
                None,
                self.bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 1 + np.arange(self.bcode_meta.n_targets),
                boost_softmax_weights = boost_softmax_weights,
                insert_nbinom_m = insert_nbinom_m,
                insert_nbinom_logit = insert_nbinom_logit,
                insert_zero_prob = insert_zero_prob)
        self._create_simulator(mdl)

        target_tract = TargetTract(0,0,0,0)
        num_replicates = 8000
        left_trims = []
        right_trims = []
        insert_lens = []
        for i in range(num_replicates):
            allele = self.allele_sim.get_root().alleles[0]
            left_trim_raw, right_trim_raw, insert_str_raw = self.allele_sim._do_repair(allele, target_tract)
            allele_events = allele.get_event_encoding()
            evt = allele_events.events[0]
            left_trim, right_trim = evt.get_trim_lens(self.bcode_meta)
            insert_len = evt.insert_len
            assert right_trim == right_trim_raw
            assert left_trim == left_trim_raw
            assert insert_len == len(insert_str_raw)
            right_trims.append(right_trim)
            left_trims.append(left_trim)
            insert_lens.append(insert_len)

        mean_nbinom = insert_nbinom_prob/(1 - insert_nbinom_prob) * insert_nbinom_m
        var_nbinom = insert_nbinom_prob/np.power(1 - insert_nbinom_prob, 2) * insert_nbinom_m
        second_moment_nbinom = var_nbinom + np.power(mean_nbinom, 2)
        insert_mean_true = 1./3 * (1 + mean_nbinom) + 2./3 * (1 - insert_zero_prob) * mean_nbinom
        insert_var_true = 1/3. * 2/3. + (1 - insert_zero_prob) * second_moment_nbinom - np.power((1 - insert_zero_prob) * mean_nbinom, 2)

        print(np.mean(insert_lens), np.var(insert_lens))
        print(insert_mean_true, insert_var_true)
        self.assertTrue(np.mean(insert_lens) < insert_mean_true + 2 * np.sqrt(insert_var_true/num_replicates))
        self.assertTrue(insert_mean_true - 2 * np.sqrt(insert_var_true/num_replicates) < np.mean(insert_lens))
        self.assertTrue(np.isclose(np.var(insert_lens), insert_var_true, atol=0.2))

    def test_neg_beta_insert_intertarg(self):
        insert_nbinom_m = np.array([1])
        insert_nbinom_logit = np.array([0])
        insert_nbinom_prob = sigmoid(insert_nbinom_logit)
        insert_zero_prob = np.array([0.1])
        boost_softmax_weights = np.array([1,1,1])
        mdl = CLTLikelihoodModel(
                None,
                self.bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 1 + np.arange(self.bcode_meta.n_targets),
                boost_softmax_weights = boost_softmax_weights,
                insert_nbinom_m = insert_nbinom_m,
                insert_nbinom_logit = insert_nbinom_logit,
                insert_zero_prob = insert_zero_prob)
        self._create_simulator(mdl)

        target_tract = TargetTract(0,0,1,1)
        num_replicates = 8000
        left_trims = []
        right_trims = []
        insert_lens = []
        for i in range(num_replicates):
            allele = self.allele_sim.get_root().alleles[0]
            left_trim_raw, right_trim_raw, insert_str_raw = self.allele_sim._do_repair(allele, target_tract)
            allele_events = allele.get_event_encoding()
            evt = allele_events.events[0]
            left_trim, right_trim = evt.get_trim_lens(self.bcode_meta)
            insert_len = evt.insert_len
            assert right_trim == right_trim_raw
            assert left_trim == left_trim_raw
            assert insert_len == len(insert_str_raw)
            right_trims.append(right_trim)
            left_trims.append(left_trim)
            insert_lens.append(insert_len)

        mean_nbinom = insert_nbinom_prob/(1 - insert_nbinom_prob) * insert_nbinom_m
        var_nbinom = insert_nbinom_prob/np.power(1 - insert_nbinom_prob, 2) * insert_nbinom_m
        second_moment_nbinom = var_nbinom + np.power(mean_nbinom, 2)
        insert_mean_true = (1 - insert_zero_prob) * mean_nbinom
        insert_var_true = (1 - insert_zero_prob) * second_moment_nbinom - np.power((1 - insert_zero_prob) * mean_nbinom, 2)

        print(np.mean(insert_lens), np.var(insert_lens))
        print(insert_mean_true, insert_var_true)
        self.assertTrue(np.mean(insert_lens) < insert_mean_true + 2 * np.sqrt(insert_var_true/num_replicates))
        self.assertTrue(insert_mean_true - 2 * np.sqrt(insert_var_true/num_replicates) < np.mean(insert_lens))
        self.assertTrue(np.isclose(np.var(insert_lens), insert_var_true, atol=0.1))

    def test_neg_beta_left_del_intertarg(self):
        trim_zero_probs = np.array([0.1,0.2,0.4,0.2])
        trim_short_nbinom_m = np.array([1,2,3,4])
        trim_short_nbinom_logits = np.array([-0.2, -0.7, 0.2, 0.8])
        mdl = CLTLikelihoodModel(
                None,
                self.bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 1 + np.arange(self.bcode_meta.n_targets),
                trim_zero_probs = trim_zero_probs,
                trim_short_nbinom_m = trim_short_nbinom_m,
                trim_short_nbinom_logits = trim_short_nbinom_logits)
        self._create_simulator(mdl)

        target_tract = TargetTract(0,0,1,1)
        num_replicates = 8000
        left_trims = []
        right_trims = []
        insert_lens = []
        for i in range(num_replicates):
            allele = self.allele_sim.get_root().alleles[0]
            left_trim_raw, right_trim_raw, insert_str_raw = self.allele_sim._do_repair(allele, target_tract)
            allele_events = allele.get_event_encoding()
            evt = allele_events.events[0]
            left_trim, right_trim = evt.get_trim_lens(self.bcode_meta)
            insert_len = evt.insert_len
            assert right_trim == right_trim_raw
            assert left_trim == left_trim_raw
            assert insert_len == len(insert_str_raw)
            right_trims.append(right_trim)
            left_trims.append(left_trim)
            insert_lens.append(insert_len)

        dist_index = 1
        trim_left_zero_prob = trim_zero_probs[dist_index]
        max_trim_len = self.bcode_meta.left_long_trim_min[0] - 1
        trim_left_dist = ZeroInflatedBoundedNegativeBinomial(
                0,
                max_trim_len,
                trim_short_nbinom_m[dist_index],
                trim_short_nbinom_logits[dist_index])
        mean_nbinom = np.sum([
            k * trim_left_dist.pmf(k) for k in range(max_trim_len + 1)])
        left_trim_mean_true = (1 - trim_left_zero_prob) * mean_nbinom
        second_moment_nbinom = np.sum([
            np.power(k, 2) * trim_left_dist.pmf(k) for k in range(max_trim_len + 1)])
        left_trim_var_true = (1 - trim_left_zero_prob) * second_moment_nbinom - np.power(mean_nbinom * (1 - trim_left_zero_prob), 2)

        print(np.mean(left_trims), np.var(left_trims))
        print(left_trim_mean_true, left_trim_var_true)
        self.assertTrue(np.mean(left_trims) < left_trim_mean_true + 2 * np.sqrt(left_trim_var_true/num_replicates))
        self.assertTrue(left_trim_mean_true - 2 * np.sqrt(left_trim_var_true/num_replicates) < np.mean(left_trims))
        self.assertTrue(np.isclose(left_trim_var_true, np.var(left_trims), atol=0.03))
