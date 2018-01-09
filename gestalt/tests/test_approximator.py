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
