import unittest
import numpy as np

from allele_simulator import AlleleSimulator
from allele import Allele

class BarcordSimulatorTestCase(unittest.TestCase):
    def setUp(self):
        self.allele = Allele()
        self.TARGET_LAM = np.array([0.5] * self.allele.n_targets)
        self.REPAIR_RATES = np.array([1] * self.allele.n_targets)
        self.allele_sim = AlleleSimulator(
            self.TARGET_LAM,
            self.REPAIR_RATES,
            indel_probability=1,
            left_del_mu=2,
            right_del_mu=2,
            insertion_mu=1)

    def test_race_process_only_cut_possible(self):
        # Nothing to repair, only one thing to cut
        for i in range(self.allele.n_targets - 1):
            self.allele.cut(i)
            self.allele.indel(i, i, 1, 0, "")
        race_winner, event_time = self.allele_sim._race_repair_target_cutting(self.allele)
        self.assertEqual(race_winner, self.allele.n_targets - 1)
        self.assertNotEqual(event_time, None)

    def test_race_process_everything_broken(self):
        # Nothing to cut since everything needs repair
        for i in range(self.allele.n_targets):
            self.allele.cut(i)
        race_winner, event_time = self.allele_sim._race_repair_target_cutting(self.allele)
        self.assertEqual(race_winner, -1)
        self.assertNotEqual(event_time, None)

    def test_race_process_no_events(self):
        # Nothing to repair/cut
        for i in range(self.allele.n_targets):
            self.allele.cut(i)
            self.allele.indel(i, i, 0, 0, "asdf")
        race_winner, event_time = self.allele_sim._race_repair_target_cutting(
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
        self.assertEqual(len(new_allele.needs_repair), 0)

    def test_repair(self):
        # After repair process, there should be no more things to repair
        self.allele.cut(0)
        self.allele.cut(self.allele.n_targets - 1)
        self.allele_sim._do_repair(self.allele)
        self.assertEqual(self.allele.needs_repair, set())
