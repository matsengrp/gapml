import unittest
import numpy as np

from barcode_simulator import BarcodeSimulator
from barcode import Barcode

class BarcordSimulatorTestCase(unittest.TestCase):
    def setUp(self):
        self.bcode = Barcode()
        self.TARGET_LAM = np.array([0.5] * self.bcode.n_targets)
        self.REPAIR_RATES = np.array([1] * self.bcode.n_targets)
        self.bcode_sim = BarcodeSimulator(
            self.TARGET_LAM,
            self.REPAIR_RATES,
            indel_probability=1,
            left_del_mu=2,
            right_del_mu=2,
            insertion_mu=1)

    def test_race_process_only_cut_possible(self):
        # Nothing to repair, only one thing to cut
        for i in range(self.bcode.n_targets - 1):
            self.bcode.cut(i)
            self.bcode.indel(i, i, 1, 0, "")
        race_winner, event_time = self.bcode_sim._race_repair_target_cutting(self.bcode)
        self.assertEqual(race_winner, self.bcode.n_targets - 1)
        self.assertNotEqual(event_time, None)

    def test_race_process_everything_broken(self):
        # Nothing to cut since everything needs repair
        for i in range(self.bcode.n_targets):
            self.bcode.cut(i)
        race_winner, event_time = self.bcode_sim._race_repair_target_cutting(self.bcode)
        self.assertEqual(race_winner, -1)
        self.assertNotEqual(event_time, None)

    def test_race_process_no_events(self):
        # Nothing to repair/cut
        for i in range(self.bcode.n_targets):
            self.bcode.cut(i)
            self.bcode.indel(i, i, 0, 0, "asdf")
        race_winner, event_time = self.bcode_sim._race_repair_target_cutting(
            self.bcode)
        self.assertEqual(race_winner, None)
        self.assertEqual(event_time, None)

    def test_simulate(self):
        old_bcode_str = str(self.bcode)

        # Simulate for zero time. Nothing should happen
        new_bcode = self.bcode_sim.simulate(self.bcode, 0)
        self.assertEqual(str(new_bcode), old_bcode_str)

        # Simulate for long time
        # (This is an assumption this bcode simulator makes!)
        new_bcode = self.bcode_sim.simulate(self.bcode, 10)
        # Make sure the old bcode is not affected
        self.assertEqual(old_bcode_str, str(self.bcode))
        # Barcode should be contiguous
        self.assertEqual(len(new_bcode.needs_repair), 0)

    def test_repair(self):
        # After repair process, there should be no more things to repair
        self.bcode.cut(0)
        self.bcode.cut(self.bcode.n_targets - 1)
        self.bcode_sim._do_repair(self.bcode)
        self.assertEqual(self.bcode.needs_repair, set())
