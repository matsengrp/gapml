from typing import List, Tuple
import numpy as np

from constants import BARCODE_V7, NUM_BARCODE_V7_TARGETS

class BarcodeMetadata:
    def __init__(self,
            unedited_barcode: List[str] = BARCODE_V7,
            num_barcodes: int = 1,
            cut_site: int = 6,
            crucial_pos_len: List[int] = [6,6]):
        """
        @param unedited_barcode: the original state of the barcode
        @param cut_sites: offset from 3' end of target for Cas9 cutting,
                        so a cut_site of 6 means that we start inserting
                        such that the inserted seq is 6 nucleotides from
                        the 3' end of the target
        @param crucial_pos_len: which positions to the left and right of the
                        cut site must not be disturbed for the target to
                        remain active
        """
        # The original barcode
        self.unedited_barcode = unedited_barcode
        self.num_barcodes = num_barcodes
        self.orig_substr_lens = [len(s) for s in unedited_barcode]
        self.orig_length = sum(self.orig_substr_lens)
        self.n_targets = (len(self.unedited_barcode) - 1) // 2

        self.cut_sites = [cut_site] * self.n_targets
        # absolute positions of cut locations
        self.abs_cut_sites = [
            sum(self.orig_substr_lens[:2 * (i + 1)]) - self.cut_sites[i]
            for i in range(self.n_targets)
        ]

        # Range of positions for each target
        # regarding which positions must be unedited
        # for this target to still be active.
        self.pos_sites = []
        for i in range(self.n_targets):
            cut_site = self.abs_cut_sites[i]
            right = min(self.orig_length - 1, cut_site + crucial_pos_len[1] - 1)
            left = max(0, cut_site - crucial_pos_len[0])
            self.pos_sites.append((left, right))

        # Min length of a long trim for target i -- left
        self.left_long_trim_min = [
            self.abs_cut_sites[i] - self.pos_sites[i - 1][1]
            for i in range(1, self.n_targets)]
        # Min length of a long trim for target i -- right
        self.right_long_trim_min = [
            self.pos_sites[i + 1][0] - self.abs_cut_sites[i] + 1 for i in range(self.n_targets - 1)]

        # Max length of any trim for target i -- left
        self.left_max_trim = [
            self.abs_cut_sites[i] - self.abs_cut_sites[i - 1] - 1 for i in range(1, self.n_targets)]
        self.left_max_trim = [self.abs_cut_sites[0]] + self.left_max_trim
        self.left_long_trim_min = [self.left_max_trim[0]] + self.left_long_trim_min

        # Max length of any trim for target i -- right
        self.right_max_trim = [
            self.abs_cut_sites[i + 1] - self.abs_cut_sites[i] for i in range(self.n_targets - 1)]
        self.right_max_trim += [self.orig_length - self.abs_cut_sites[-1]]
        self.right_long_trim_min += [self.right_max_trim[-1]]

    @staticmethod
    def create_fake_barcode_str(num_targets: int):
        """
        @return List[str], return a single barcode, not an object!
        """
        start_end_spacer = BARCODE_V7[0]
        target_seq = BARCODE_V7[1]
        btw_target_spacer = BARCODE_V7[2]
        bcode = (start_end_spacer, target_seq)
        for i in range(num_targets - 1):
            bcode += (btw_target_spacer, target_seq)
        bcode += (start_end_spacer,)
        return bcode
