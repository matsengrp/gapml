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
from bounded_distributions import ConditionalBoundedNegativeBinomial, ShiftedNegativeBinomial
from bounded_distributions import ConditionalBoundedPoisson


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
                target_lams = 1 + np.arange(self.bcode_meta.n_targets),
                use_poisson = False)
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
                target_lams = 1 + np.arange(self.bcode_meta.n_targets),
                use_poisson=False)
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

    def test_neg_beta_right_del_focal(self):
        """
        Check trim and insertion lengths match up when focal cut with short trims
        """
        trim_zero_probs = np.array([0.1,0.2,0.4,0.2])
        trim_zero_probs_reshape = trim_zero_probs.reshape([2,-1])
        trim_short_params = np.array([1,0.3,1,0.1])
        trim_short_params_reshape = trim_short_params.reshape([2,-1])
        insert_zero_prob = np.array([0.4])
        insert_params = np.array([0.5,0.5])
        mdl = CLTLikelihoodModel(
                None,
                self.bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 1 + np.arange(self.bcode_meta.n_targets),
                trim_zero_probs = trim_zero_probs,
                trim_short_params = trim_short_params,
                insert_zero_prob = insert_zero_prob,
                insert_params = insert_params,
                use_poisson = False)
        self._create_simulator(mdl)

        target_tract = TargetTract(1,1,1,1)
        num_replicates = 2000
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

        # Normalization term is for having some allele effect
        normalization = 1 - (trim_zero_probs_reshape[0,0] * trim_zero_probs_reshape[1,0] * insert_zero_prob)

        # Test left
        dist_index = 0
        trim_left_zero_prob = trim_zero_probs_reshape[dist_index, 0]
        max_trim_len = self.bcode_meta.left_long_trim_min[1] - 1
        trim_left_dist = ConditionalBoundedNegativeBinomial(
                1,
                max_trim_len,
                np.exp(trim_short_params_reshape[dist_index,0]),
                trim_short_params_reshape[dist_index,1])
        left_trim_mean_true = (1 - trim_left_zero_prob) * np.sum([k * trim_left_dist.pmf(k) for k in range(1, max_trim_len + 1)])/normalization

        # Check prob distribution sums to 1
        self.assertTrue(np.isclose(1, np.sum([trim_left_dist.pmf(k) for k in range(1, max_trim_len + 1)])))
        # Check the means match
        self.assertTrue(np.mean(left_trims) + 2 * np.sqrt(np.var(left_trims)/num_replicates) > left_trim_mean_true)

        # Test right
        dist_index = 1
        trim_right_zero_prob = trim_zero_probs_reshape[dist_index, 0]
        max_trim_len = self.bcode_meta.right_long_trim_min[1] - 1
        trim_right_dist = ConditionalBoundedNegativeBinomial(
                1,
                max_trim_len,
                np.exp(trim_short_params_reshape[dist_index,0]),
                trim_short_params_reshape[dist_index,1])
        right_trim_mean_true = (1 - trim_right_zero_prob) * np.sum([k * trim_right_dist.pmf(k) for k in range(1, max_trim_len + 1)])/normalization

        # Check prob distribution sums to 1
        self.assertTrue(np.isclose(1, np.sum([trim_right_dist.pmf(k) for k in range(1, max_trim_len + 1)])))
        # Check the means match
        self.assertTrue(np.mean(right_trims) + 2 * np.sqrt(np.var(right_trims)/num_replicates) > right_trim_mean_true)

        # Test insertion len
        insert_dist = ShiftedNegativeBinomial(
                1,
                np.exp(insert_params[0]),
                insert_params[1])
        insert_mean_true = (1 - insert_zero_prob) * np.sum([k * insert_dist.pmf(k) for k in range(1, 10000)])/normalization

        # Check prob distribution sums to 1
        self.assertTrue(np.isclose(1, np.sum([insert_dist.pmf(k) for k in range(1, 1000000)])))
        # Check the means match
        self.assertTrue(np.mean(insert_lens) + 2 * np.sqrt(np.var(insert_lens)/num_replicates) > insert_mean_true)

    def test_neg_beta_right_del_long(self):
        """
        Check trim and insertion lengths match up when focal cut with long trims
        """
        trim_zero_probs = np.array([0.1,0.2,0.4,0.2])
        trim_zero_probs_reshape = trim_zero_probs.reshape([2,-1])
        trim_long_params = np.array([0.1,0.3])
        trim_long_params_reshape = trim_long_params.reshape([2,-1])
        insert_zero_prob = np.array([0.4])
        insert_params = np.array([0.5,0.5])
        mdl = CLTLikelihoodModel(
                None,
                self.bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 1 + np.arange(self.bcode_meta.n_targets),
                trim_zero_probs = trim_zero_probs,
                trim_long_params = trim_long_params,
                insert_zero_prob = insert_zero_prob,
                insert_params = insert_params,
                use_poisson = False)
        self._create_simulator(mdl)

        target_tract = TargetTract(0,1,1,2)
        num_replicates = 2000
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
            self.assertTrue(right_trim >= self.bcode_meta.right_long_trim_min[1])
            self.assertTrue(left_trim >= self.bcode_meta.left_long_trim_min[1])
            self.assertTrue(right_trim == right_trim_raw)
            self.assertTrue(left_trim == left_trim_raw)
            self.assertTrue(insert_len == len(insert_str_raw))
            right_trims.append(right_trim)
            left_trims.append(left_trim)
            insert_lens.append(insert_len)

        # Test left
        dist_index = 0
        trim_left_dist = ConditionalBoundedPoisson(
                self.bcode_meta.left_long_trim_min[1],
                self.bcode_meta.left_max_trim[1],
                np.exp(trim_long_params_reshape[dist_index,0]))
        left_trim_range = range(self.bcode_meta.left_long_trim_min[1], self.bcode_meta.left_max_trim[1] + 1)
        left_trim_mean_true = np.sum([k * trim_left_dist.pmf(k) for k in left_trim_range])

        # Check prob distribution sums to 1
        self.assertTrue(np.isclose(1, np.sum([trim_left_dist.pmf(k) for k in left_trim_range])))
        # Check the means match
        self.assertTrue(np.mean(left_trims) + 2 * np.sqrt(np.var(left_trims)/num_replicates) > left_trim_mean_true)

        # Test right
        dist_index = 1
        trim_right_dist = ConditionalBoundedPoisson(
                self.bcode_meta.right_long_trim_min[1],
                self.bcode_meta.right_max_trim[1],
                np.exp(trim_long_params_reshape[dist_index,0]))
        right_trim_range = range(self.bcode_meta.right_long_trim_min[1], self.bcode_meta.right_max_trim[1] + 1)
        right_trim_mean_true = np.sum([k * trim_right_dist.pmf(k) for k in right_trim_range])

        # Check prob distribution sums to 1
        self.assertTrue(np.isclose(1, np.sum([trim_right_dist.pmf(k) for k in right_trim_range])))
        # Check the means match
        self.assertTrue(np.mean(right_trims) + 2 * np.sqrt(np.var(right_trims)/num_replicates) > right_trim_mean_true)

        # Test insertion len
        insert_dist = ShiftedNegativeBinomial(
                1,
                np.exp(insert_params[0]),
                insert_params[1])
        insert_mean_true = (1 - insert_zero_prob) * np.sum([k * insert_dist.pmf(k) for k in range(1, 10000)])

        # Check prob distribution sums to 1
        self.assertTrue(np.isclose(1, np.sum([insert_dist.pmf(k) for k in range(1, 1000000)])))
        # Check the means match
        self.assertTrue(np.mean(insert_lens) + 2 * np.sqrt(np.var(insert_lens)/num_replicates) > insert_mean_true)
