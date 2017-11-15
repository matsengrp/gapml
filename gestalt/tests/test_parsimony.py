import unittest

from parsimony_solver import MaxEventSolver
from barcode_events import Event, BarcodeEvents

class ParsimonyTestCase(unittest.TestCase):
    def setUp(self):
        self.solver = MaxEventSolver([1 + i * 5 for i in range(9)])

    def test_none(self):
        e1s = Event(start_pos=1, del_len=2, insert_str="", min_target=0, max_target=0)
        b1 = BarcodeEvents([e1s])
        b2 = BarcodeEvents([])
        pars_bcode = self.solver.get_parsimony_barcode([b1, b2])
        self.assertEqual(pars_bcode.events, [])

        e1s = Event(start_pos=1, del_len=2, insert_str="", min_target=0, max_target=0)
        b1 = BarcodeEvents([e1s])
        e2s  = Event(start_pos=10, del_len=20, insert_str="", min_target=1, max_target=1)
        b2 = BarcodeEvents([e2s])
        pars_bcode = self.solver.get_parsimony_barcode([b1, b2])
        self.assertEqual(pars_bcode.events, [])

        e1s = Event(start_pos=1, del_len=2, insert_str="atcg", min_target=0, max_target=0)
        b1 = BarcodeEvents([e1s])
        e2s = Event(start_pos=1, del_len=2, insert_str="ac", min_target=0, max_target=0)
        b2 = BarcodeEvents([e2s])
        pars_bcode = self.solver.get_parsimony_barcode([b1, b2])
        self.assertEqual(pars_bcode.events, [])

    def test_zero(self):
        e0 = Event(start_pos=0, del_len=2, insert_str="", min_target=0, max_target=0)
        e1 = Event(start_pos=5, del_len=2, insert_str="", min_target=1, max_target=1)
        e2 = Event(start_pos=10, del_len=2, insert_str="", min_target=2, max_target=2)
        e3 = Event(start_pos=15, del_len=2, insert_str="", min_target=3, max_target=3)
        b1 = BarcodeEvents([e1])
        b2 = BarcodeEvents([e1])
        pars_bcode = self.solver.get_parsimony_barcode([b1, b2])
        self.assertEqual(pars_bcode.events, b1.events)

        b1 = BarcodeEvents([e1, e2])
        b2 = BarcodeEvents([e0, e1, e3])
        pars_bcode = self.solver.get_parsimony_barcode([b1, b2])
        self.assertEqual(pars_bcode.events, [e1])

    def test_one(self):
        e0 = Event(start_pos=0, del_len=2, insert_str="", min_target=0, max_target=0)
        e1 = Event(start_pos=0, del_len=11, insert_str="", min_target=0, max_target=2)
        e2 = Event(start_pos=0, del_len=21, insert_str="", min_target=0, max_target=4)
        e3 = Event(start_pos=8, del_len=13, insert_str="", min_target=1, max_target=4)
        e4 = Event(start_pos=0, del_len=2, insert_str="", min_target=0, max_target=0)
        e5 = Event(start_pos=0, del_len=41, insert_str="", min_target=0, max_target=8)
        b0 = BarcodeEvents([e0])
        b1 = BarcodeEvents([e1])
        b2 = BarcodeEvents([e2])

        # check matching at the start pos
        pars_bcode = self.solver.get_parsimony_barcode([b1, b2, b0])
        self.assertEqual(pars_bcode.events, [e0])

        # check matching at the end pos
        b3 = BarcodeEvents([e3])
        pars_bcode = self.solver.get_parsimony_barcode([b2, b3])
        self.assertEqual(
            [str(i) for i in pars_bcode.events], [str(e3)])

        # either of the two targets can explain
        b4 = BarcodeEvents([e4, e3])
        pars_bcode = self.solver.get_parsimony_barcode([b2, b4])
        self.assertEqual(
            [str(i) for i in pars_bcode.events],
            [str(e4), str(e3)])

        # single long inter-target can explain many events
        b5 = BarcodeEvents([e5])
        pars_bcode = self.solver.get_parsimony_barcode([b5, b4])
        self.assertEqual(pars_bcode.events, b4.events)

    def test_two(self):
        e0 = Event(start_pos=0, del_len=7, insert_str="", min_target=0, max_target=1)
        e1 = Event(start_pos=1, del_len=11, insert_str="", min_target=0, max_target=2)
        e3 = Event(start_pos=4, del_len=13, insert_str="", min_target=1, max_target=2)
        e4 = Event(start_pos=19, del_len=10, insert_str="", min_target=3, max_target=5)
        e5 = Event(start_pos=10, del_len=11, insert_str="", min_target=2, max_target=3)
        e6 = Event(start_pos=10, del_len=11, insert_str="actg", min_target=2, max_target=3)
        b0 = BarcodeEvents([e0])
        b1 = BarcodeEvents([e1])

        pars_bcode = self.solver.get_parsimony_barcode([b0, b1])
        self.assertEqual(
            [str(i) for i in pars_bcode.events],
            [str(Event(start_pos=2, del_len=4, min_target=1, max_target=1, insert_str="*"))])

        b3 = BarcodeEvents([e3, e4])
        b4 = BarcodeEvents([e5])
        pars_bcode = self.solver.get_parsimony_barcode([b4, b3])
        self.assertEqual(
            [str(i) for i in pars_bcode.events],
            [
                str(Event(start_pos=10, del_len=6, min_target=2, max_target=2, insert_str="*")),
                str(Event(start_pos=19, del_len=1, min_target=3, max_target=3, insert_str="*"))
            ])

        b6 = BarcodeEvents([e6])
        pars_bcode = self.solver.get_parsimony_barcode([b6, b4])
        self.assertEqual(len(pars_bcode.events), 1)
        self.assertEqual(pars_bcode.events[0].min_target, pars_bcode.events[0].max_target)
