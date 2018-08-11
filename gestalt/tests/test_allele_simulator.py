import unittest
import numpy as np
import tensorflow as tf

from target_status import TargetStatus, TargetDeactTract
from allele_simulator_simult import AlleleSimulatorSimultaneous
from allele import Allele
from barcode_metadata import BarcodeMetadata
from clt_likelihood_model import CLTLikelihoodModel
from optim_settings import KnownModelParams

class AlleleSimulatorTestCase(unittest.TestCase):
    def setUp(self):
        bcode_meta = BarcodeMetadata()
        self.sess = tf.InteractiveSession()
        self.known_params = KnownModelParams(tot_time=True)
        self.mdl = CLTLikelihoodModel(
                None,
                bcode_meta,
                self.sess,
                known_params = self.known_params,
                target_lams = 1 + np.arange(bcode_meta.n_targets))
        tf.global_variables_initializer().run()

        self.allele_sim = AlleleSimulatorSimultaneous(
                self.mdl,
                boost_probs = .3 * np.ones(3))
        self.allele = self.allele_sim.get_root().alleles[0]

    def test_race_process_no_events(self):
        # Nothing to repair/cut
        for i in range(self.allele.bcode_meta.n_targets):
            self.allele.indel(i, i, 0, 0, "att")
        race_winner, event_time = self.allele_sim._race_target_tracts(
            self.allele)
        self.assertEqual(race_winner, None)
        self.assertEqual(event_time, None)

    def test_simulate(self):
        old_allele_str = str(self.allele)

        # Simulate for zero time. Nothing should happen
        new_allele = self.allele_sim.simulate(self.allele, 0)
        self.assertEqual(str(new_allele), old_allele_str)

        # Simulate for long time
        # (This is an assumption this allele simulator makes!)
        new_allele = self.allele_sim.simulate(self.allele, 10)
        # Make sure the old allele is not affected
        self.assertEqual(old_allele_str, str(self.allele))
        # Allele should be contiguous
        self.assertEqual(new_allele.get_target_status(), TargetStatus(TargetDeactTract(0,9)))
