import unittest

from indel_sets import *
from approximator import ApproximatorLB

class ApproximatorLBTestCase(unittest.TestCase):
    def test_partition(self):
        tts = (TargetTract(8,8,8,8),)
        anc_state = AncState([
            Wildcard(1,4),
            Wildcard(8,8)])
        parts = ApproximatorLB.partition(tts, anc_state)
        self.assertEqual(parts[Wildcard(1,4)], ())
        self.assertEqual(parts[Wildcard(8,8)], (tts[0],))

        tts = (
            TargetTract(1,1,2,2),
            TargetTract(3,3,3,4),
            TargetTract(8,8,8,8))
        anc_state = AncState([
            Wildcard(1,4),
            Wildcard(8,9)])
        parts = ApproximatorLB.partition(tts, anc_state)
        self.assertEqual(parts[Wildcard(1,4)], (tts[0], tts[1]))
        self.assertEqual(parts[Wildcard(8,9)], (tts[2],))

        tts = ()
        anc_state = AncState([
            Wildcard(1,4),
            Wildcard(8,9)])
        parts = ApproximatorLB.partition(tts, anc_state)
        self.assertEqual(parts[Wildcard(1,4)], ())
        self.assertEqual(parts[Wildcard(8,9)], ())

    def test_deactivated_targets(self):
        tt_grp = (TargetTract(0,0,1,2), TargetTract(8,8,8,8))
        deact_targs = ApproximatorLB.get_deactivated_targets(tt_grp)
        self.assertEqual(deact_targs, set([0,1,2,8]))

        tt_grp = ()
        deact_targs = ApproximatorLB.get_deactivated_targets(tt_grp)
        self.assertEqual(deact_targs, set())

        tt_grp = (TargetTract(0,1,8,9),)
        deact_targs = ApproximatorLB.get_deactivated_targets(tt_grp)
        self.assertEqual(deact_targs, set(range(10)))

    def test_active_any_trim_targets(self):
        tt_grp = (TargetTract(1,1,1,2), TargetTract(4,4,4,4))
        indel_set = Wildcard(0,8)
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [0,3,5,6,7,8])

        tt_grp = (TargetTract(2,2,3,3), TargetTract(4,4,4,4))
        indel_set = Wildcard(0,5)
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [0,1,5])

        tt_grp = (TargetTract(2,2,3,3),)
        indel_set = SingletonWC(30, 20, 2, 2, 3, 3, "hello")
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [])

        tt_grp = (TargetTract(2,2,2,2),)
        indel_set = SingletonWC(30, 20, 2, 2, 3, 3, "hello")
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [])

        tt_grp = ()
        indel_set = SingletonWC(30, 20, 2, 2, 4, 4, "hello")
        active_any_targs = ApproximatorLB.get_active_any_trim_targets(indel_set, tt_grp)
        self.assertEqual(active_any_targs, [3])

    def test_any_trim_target_tracts(self):
        active_any_targs = [0,1]
        any_tts = ApproximatorLB.get_possible_any_trim_target_tracts(active_any_targs)
        self.assertEqual(any_tts, set([
            TargetTract(0,0,0,0),
            TargetTract(0,0,0,1),
            TargetTract(0,0,1,1),
            TargetTract(0,1,1,1),
            TargetTract(1,1,1,1)]))

        active_any_targs = [0,5]
        any_tts = ApproximatorLB.get_possible_any_trim_target_tracts(active_any_targs)
        self.assertEqual(any_tts, set([
            TargetTract(0,0,0,0),
            TargetTract(0,0,5,5),
            TargetTract(5,5,5,5)]))
