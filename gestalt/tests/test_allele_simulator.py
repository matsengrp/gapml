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

    @unittest.skip("adsf")
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

    @unittest.skip("adsf")
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

    def test_neg_beta_insert(self):
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
        num_replicates = 5000
        left_trims = []
        right_trims = []
        insert_lens = []
        for i in range(num_replicates):
            allele = self.allele_sim.get_root().alleles[0]
            self.allele_sim._do_repair(allele, target_tract)
            allele_events = allele.get_event_encoding()
            evt = allele_events.events[0]
            left_trim, right_trim = evt.get_trim_lens(self.bcode_meta)
            insert_len = evt.insert_len
            right_trims.append(right_trim)
            left_trims.append(left_trim)
            insert_lens.append(insert_len)

        mean_nbinom = insert_nbinom_prob/(1 - insert_nbinom_prob) * insert_nbinom_m
        var_nbinom = insert_nbinom_prob/np.power(1 - insert_nbinom_prob, 2) * insert_nbinom_m
        second_moment_nbinom = var_nbinom + np.power(mean_nbinom, 2)
        insert_mean_true = 1./3 * 1 + (1 - insert_zero_prob) * mean_nbinom
        insert_var_true = 1/3. * 2/3. + (1 - insert_zero_prob) * second_moment_nbinom - np.power((1 - insert_zero_prob) * mean_nbinom, 2)

        print(np.mean(insert_lens), np.var(insert_lens))
        print(insert_mean_true, insert_var_true)
        self.assertTrue(np.isclose(np.mean(insert_lens), insert_mean_true, atol=0.1))
        self.assertTrue(np.isclose(np.var(insert_lens), insert_var_true, atol=0.1))
