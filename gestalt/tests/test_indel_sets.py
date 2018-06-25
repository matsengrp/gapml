import unittest

from indel_sets import *
from anc_state import AncState

class AlleleTestCase(unittest.TestCase):
    def test_intersection(self):
        w = Wildcard(2,4)
        s = SingletonWC(1,10,1,1,1,1,"Asdf")
        self.assertEqual(IndelSet.intersect(w,s), None)

        w = Wildcard(2,4)
        s = SingletonWC(1,10,2,2,4,4,"Asdf")
        self.assertEqual(IndelSet.intersect(w,s), s)

        w = Wildcard(2,4)
        s = SingletonWC(1,10,3,3,3,3,"Asdf")
        self.assertEqual(IndelSet.intersect(w,s), s)

        w = Wildcard(2,4)
        s = SingletonWC(1,10,2,3,3,4,"Asdf")
        self.assertEqual(IndelSet.intersect(w,s), s)

        w = Wildcard(2,4)
        s = SingletonWC(1,10,2,3,4,5,"Asdf")
        self.assertEqual(IndelSet.intersect(w,s), None)

    def test_intersect_ancstate(self):
        l1 = [Wildcard(1,1), SingletonWC(10,30,2, 2,2,2, "asdf"), SingletonWC(70, 100, 3,4,4,5,"")]
        l2 = [Wildcard(2,3), Wildcard(3,5)]
        anc_state1 = AncState(l1)
        anc_state2 = AncState(l2)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 2)
        self.assertTrue(l1[-1] in par_anc_state.indel_set_list)
        self.assertTrue(l1[1] in par_anc_state.indel_set_list)

        l3 = [Wildcard(4,10)]
        anc_state3 = AncState(l3)
        par_anc_state = AncState.intersect(anc_state3, anc_state2)
        self.assertEqual(par_anc_state.indel_set_list, [Wildcard(4,5)])

        l4 = [Wildcard(2,6)]
        anc_state4 = AncState(l4)
        par_anc_state = AncState.intersect(anc_state1, anc_state4)
        self.assertEqual(len(par_anc_state.indel_set_list), 2)
        self.assertTrue(l1[-1] in par_anc_state.indel_set_list)
        self.assertTrue(l1[1] in par_anc_state.indel_set_list)

        l5 = [Wildcard(8,10)]
        anc_state5 = AncState(l5)
        par_anc_state = AncState.intersect(anc_state5, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 0)

        l6 = [SingletonWC(10,30,2, 2,2,2, "a")]
        anc_state6 = AncState(l6)
        par_anc_state = AncState.intersect(anc_state6, anc_state1)
        self.assertEqual(len(par_anc_state.indel_set_list), 0)

        l7 = [l1[1]]
        anc_state7 = AncState(l7)
        par_anc_state = AncState.intersect(anc_state7, anc_state1)
        self.assertEqual(par_anc_state.indel_set_list, l7)
