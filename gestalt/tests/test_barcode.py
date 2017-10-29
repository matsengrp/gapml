import unittest

from barcode import Barcode

class BarcodeTestCase(unittest.TestCase):
    def setUp(self):
        self.NUM_TARGETS = 4
        self.ORIG_BARCODE = [
                "A",
                "TATCTT",
                "TAT",
                "TACCAT",
                "ACG",
                "GGCGAG",
                "ATG",
                "GTTGAG",
                "C"]
        self.TARGET_LEN = len(self.ORIG_BARCODE[1])
        self.SEP_LEN = len(self.ORIG_BARCODE[2])
        self.CUT_SITE_NUM = 3
        self.CUT_SITES = [self.CUT_SITE_NUM] * self.NUM_TARGETS
        self.barcode = Barcode(
            self.ORIG_BARCODE,
            self.ORIG_BARCODE,
            self.CUT_SITES)

    def test_active_targets(self):
        active_targets = self.barcode.get_active_targets()
        self.assertEqual(
            set(active_targets),
            set(range(self.NUM_TARGETS)))

        self.barcode.cut(0)
        active_targets = self.barcode.get_active_targets()
        self.assertEqual(
            set(active_targets),
            set(range(1,self.NUM_TARGETS)))

        self.barcode.indel(0, 0, 0, 0, "")
        active_targets = self.barcode.get_active_targets()
        self.assertEqual(
            set(active_targets),
            set(range(self.NUM_TARGETS)))

        self.barcode.cut(0)
        self.barcode.indel(0, 0, 0, 0, "atcg")
        active_targets = self.barcode.get_active_targets()
        self.assertEqual(
            set(active_targets),
            set(range(1, self.NUM_TARGETS)))

    def test_indel(self):
        self.barcode.cut(0)
        self.assertEqual(self.barcode.needs_repair, set([0]))
        self.barcode.indel(0, 0, 1, 2, "atcg")
        barcode_str = str(self.barcode)
        true_prefix = "%s%s-atcg--%s" % (
            self.ORIG_BARCODE[0],
            self.ORIG_BARCODE[1][:self.TARGET_LEN - self.CUT_SITE_NUM - 1],
            self.ORIG_BARCODE[1][-self.CUT_SITE_NUM + 2:])
        self.assertTrue(barcode_str.startswith(true_prefix))
        self.assertTrue(len(self.barcode.needs_repair) == 0)

        self.barcode.cut(1)
        self.barcode.cut(3)
        self.assertEqual(self.barcode.needs_repair, set([1,3]))
        # cut off 4 to the left, which cuts thru to a separator sequence
        # cut off 6 to the right, though the sequence actually only has 4 more to the right
        self.barcode.indel(1, 3, 4, 6, "atcg")
        barcode_str = str(self.barcode)
        true_suffix = "%s%satcg%s" % (
            self.ORIG_BARCODE[2][:-1],
            "-" * 4,
            "-" * (4 + (self.TARGET_LEN + self.SEP_LEN) * 2))
        self.assertTrue(barcode_str.endswith(true_suffix))
        self.assertTrue(len(set(self.barcode.needs_repair)) == 0)

    def test_events(self):
        evts = self.barcode.get_events()
        self.assertEqual(len(evts), 0)

        self.barcode.cut(0)
        self.barcode.indel(0, 0, 1, 2, "atcg")
        evts = self.barcode.get_events()
        self.assertEqual(evts, [(3, 6, "atcg")])

    def test_process_events(self):
        evts = [(0, 6, "atcg"), (10, 27, ""), (35, 35, "atgc")]
        self.barcode.process_events(evts)
        self.assertTrue(self.barcode.get_events(), evts)

