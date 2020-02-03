import unittest

from barcode_metadata import BarcodeMetadata
from target_status import TargetStatus, TargetDeactTract
from indel_sets import TargetTract

class CollapsedTreeTestCase(unittest.TestCase):
    def setUp(self):
        self.bcode_meta = BarcodeMetadata()

    def test_properties(self):
        deact_tracts = [
                TargetDeactTract(0,1),
                TargetDeactTract(3,3),
                TargetDeactTract(5,7)]
        targ_stat = TargetStatus(*deact_tracts)
        self.assertEqual(
                targ_stat.deact_targets,
                [0,1,3,5,6,7])

    def test_merge(self):
        deact_tracts1 = [
                TargetDeactTract(0,1),
                TargetDeactTract(3,3),
                TargetDeactTract(5,7)]
        targ_stat1 = TargetStatus(*deact_tracts1)

        deact_tracts2 = [TargetDeactTract(9,9)]
        targ_stat2 = TargetStatus(*deact_tracts2)

        merged_targ_stat = targ_stat1.merge(targ_stat2)
        self.assertEqual(
                merged_targ_stat,
                TargetStatus(*(deact_tracts1 + deact_tracts2)))

    def test_minus(self):
        deact_tracts1 = [TargetDeactTract(0,1)]
        targ_stat1 = TargetStatus(*deact_tracts1)

        self.assertEqual(
                targ_stat1.minus(TargetStatus()),
                set([0,1]))

        deact_tracts2 = [TargetDeactTract(0,1), TargetDeactTract(3,3)]
        targ_stat2 = TargetStatus(*deact_tracts2)

        self.assertEqual(
                targ_stat2.minus(TargetStatus(TargetDeactTract(0,1))),
                set([3]))

        deact_tracts3 = [TargetDeactTract(3,6)]
        targ_stat3 = TargetStatus(*deact_tracts3)

        self.assertEqual(
                targ_stat3.minus(targ_stat2),
                set())

    def test_active_targets(self):
        deact_tracts1 = [TargetDeactTract(0,1), TargetDeactTract(4,4)]
        targ_stat1 = TargetStatus(*deact_tracts1)
        active_targets = targ_stat1.get_active_targets(self.bcode_meta)
        self.assertEqual(
                active_targets,
                [2,3,5,6,7,8,9])

    def test_possible_target_tracts(self):
        targ_stat = TargetStatus()
        bcode_meta = BarcodeMetadata(
            unedited_barcode = ("AA", "ATCGATCG", "ACTG", "ATCGATCG", "ACTG", "TGACTAGC", "TT"),
            cut_site = 3,
            crucial_pos_len = [3,3])
        target_tracts = set(targ_stat.get_possible_target_tracts(bcode_meta))
        self.assertTrue(TargetTract(0,0,2,2) in target_tracts)

        deact_tracts = [TargetDeactTract(0,2), TargetDeactTract(4,6), TargetDeactTract(8,9)]
        targ_stat = TargetStatus(*deact_tracts)
        target_tracts = set(targ_stat.get_possible_target_tracts(self.bcode_meta))
        self.assertTrue(TargetTract(2,3,3,4) in target_tracts)
        self.assertTrue(TargetTract(3,3,3,3) in target_tracts)
        self.assertTrue(TargetTract(3,3,7,8) in target_tracts)
        self.assertTrue(TargetTract(2,3,7,7) in target_tracts)
        self.assertTrue(TargetTract(6,7,7,8) in target_tracts)

    def test_get_contained_target_statuses(self):
        target_statuses = TargetDeactTract(0,1).get_contained_target_statuses()
        self.assertTrue(TargetStatus() in target_statuses)
        self.assertTrue(TargetStatus(TargetDeactTract(0,0)) in target_statuses)
        self.assertTrue(TargetStatus(TargetDeactTract(1,1)) in target_statuses)
        self.assertTrue(TargetStatus(TargetDeactTract(0,1)) in target_statuses)
        self.assertTrue(len(target_statuses) == 4)

    def test_get_all_transitions(self):
        bcode_meta = BarcodeMetadata(
            unedited_barcode = ("AA", "ATCGATCG", "ACTG", "ATCGATCG", "ACTG", "TGACTAGC", "TT"),
            cut_site = 3,
            crucial_pos_len = [3,3])
        all_transitions, _ = TargetStatus.get_all_transitions(bcode_meta)
        self.assertEqual(len(all_transitions.keys()), 8)

        self.assertEqual(all_transitions[TargetStatus(TargetDeactTract(0,2))], {})
        self.assertEqual(set(all_transitions[TargetStatus()].keys()),
                set([TargetStatus(TargetDeactTract(0,0)),
                    TargetStatus(TargetDeactTract(0,1)),
                    TargetStatus(TargetDeactTract(1,1)),
                    TargetStatus(TargetDeactTract(1,2)),
                    TargetStatus(TargetDeactTract(0,2)),
                    TargetStatus(TargetDeactTract(2,2))]))
        self.assertEqual(
                set(all_transitions[TargetStatus()][TargetStatus(TargetDeactTract(0,1))]),
                set([TargetTract(0,0,1,1),
                    TargetTract(0,0,0,1),
                    TargetTract(0,1,1,1)]))
        self.assertEqual(
                set(all_transitions[TargetStatus()][TargetStatus(TargetDeactTract(0,2))]),
                set([
                    TargetTract(0,0,2,2),
                    TargetTract(0,0,1,2),
                    TargetTract(0,1,1,2),
                    TargetTract(0,1,2,2)]))
        self.assertEqual(
                set(all_transitions[TargetStatus(TargetDeactTract(0,1))][TargetStatus(TargetDeactTract(0,2))]),
                set([
                    TargetTract(2,2,2,2),
                    TargetTract(1,2,2,2)]))
        self.assertEqual(
                set(all_transitions[TargetStatus(TargetDeactTract(1,1))][TargetStatus(TargetDeactTract(0,2))]),
                set([TargetTract(0,0,2,2)]))
