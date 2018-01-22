import unittest

from indel_sets import *

class AlleleTestCase(unittest.TestCase):
    def test_intersection(self):
        w = Wildcard(2,4)
        s = SingletonWC(1,10,1,1,"Asdf")
        self.assertEqual(IndelSet.intersect(w,s), None)

        w = Wildcard(2,4)
        s = SingletonWC(1,10,2,4,"Asdf")
        self.assertEqual(IndelSet.intersect(w,s), Wildcard(3,3))

    def test_intersect_ancstate(self):
        l1 = [Wildcard(1,1), SingletonWC(10,30, 2,2, "asdf"), SingletonWC(70, 100, 3,5,"")]
        l2 = [Wildcard(2,3), Wildcard(3,5)]
        anc_state1 = AncState(l1)
        anc_state2 = AncState(l2)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 1)
        self.assertEqual(par_anc_state.indel_set_list[0], Wildcard(4,4))

        l1 = [Wildcard(5,10)]
        anc_state1 = AncState(l1)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 1)
        self.assertEqual(par_anc_state.indel_set_list[0], Wildcard(5,5))
