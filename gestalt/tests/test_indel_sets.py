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
        l2 = [Wildcard(2,2), Wildcard(3,5)]
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

    def test_intersect_ancstate_offset_indel_sets(self):
        """
        Slightly trickier anc states with indel sets that are overlapping in weird ways
        """
        l1 = [SingletonWC(17, 29, 0, 0, 1, 1, 'tggg'), SingletonWC(73, 116, 2, 2, 6, 6, 'aggcga')]
        l2 = [SingletonWC(42, 64, 1, 1, 3, 3, 'ac'), SingletonWC(153, 2, 5, 5, 5, 5, 'a')]
        anc_state1 = AncState(l1)
        anc_state2 = AncState(l2)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 1)
        self.assertTrue(l2[-1] in par_anc_state.indel_set_list)

        l1 = [SingletonWC(17, 49, 0, 0, 2, 2, 'tggg')]
        l2 = [SingletonWC(42, 64, 1, 1, 3, 3, 'ac')]
        anc_state1 = AncState(l1)
        anc_state2 = AncState(l2)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 0)

        l1 = [SingletonWC(17, 60, 0, 0, 3, 4, '')]
        l2 = [SingletonWC(100, 34, 2, 3, 3, 3, 'ac')]
        anc_state1 = AncState(l1)
        anc_state2 = AncState(l2)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 0)

    def test_intersect_ancstate_touching_sgwcs(self):
        """
        Test annoying cases where the long trims of singleton-wildcards run into each other
        Then the preceding allele can only be correctly calculated with a post-processing step
        """
        # even though they share the singleton-wc[1,2,2,2], they can't actually be preceded by such
        # an allele
        l1 = [SingletonWC(17, 29, 0, 0, 1, 1, 'tggg'), SingletonWC(55, 35, 1, 2, 2, 2, 'a')]
        l2 = [SingletonWC(20, 4, 0, 0, 0, 0, 'ac'), SingletonWC(55, 35, 1, 2, 2, 2, 'a')]
        anc_state1 = AncState(l1)
        anc_state2 = AncState(l2)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 0)

        l1 = [SingletonWC(17, 60, 0, 0, 2, 2, 'tggg'), SingletonWC(85, 35, 2, 3, 3, 3, 'a')]
        l2 = [SingletonWC(20, 100, 0, 0, 5, 5, 'ac')]
        anc_state1 = AncState(l1)
        anc_state2 = AncState(l2)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 1)
        self.assertTrue(Wildcard(1,1) in par_anc_state.indel_set_list)

        l1 = [SingletonWC(17, 45, 0, 0, 1, 2, 'tggg'), SingletonWC(70, 3, 2, 2, 2, 2, '')]
        l2 = [SingletonWC(17, 45, 0, 0, 1, 2, 'tggg')]
        anc_state1 = AncState(l1)
        anc_state2 = AncState(l2)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 0)

        l1 = [SingletonWC(17, 45, 0, 0, 1, 2, 'tggg'), SingletonWC(70, 3, 2, 2, 2, 2, '')]
        l2 = [SingletonWC(19, 43, 0, 0, 1, 2, ''), SingletonWC(70, 3, 2, 2, 2, 2, '')]
        anc_state1 = AncState(l1)
        anc_state2 = AncState(l2)
        par_anc_state = AncState.intersect(anc_state1, anc_state2)
        self.assertEqual(len(par_anc_state.indel_set_list), 1)
        self.assertTrue(l2[-1] in par_anc_state.indel_set_list)
