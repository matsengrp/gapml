from typing import List, Tuple
import numpy as np

from barcode_events import Event
from constants import BARCODE_V7, NUM_BARCODE_V7_TARGETS

class BarcodeMetadata:
    def __init__(self,
            unedited_barcode: List[str] = BARCODE_V7,
            cut_sites: List[int] = [6] * NUM_BARCODE_V7_TARGETS):
        """
        @param unedited_barcode: the original state of the barcode
        @param cut_sites: offset from 3' end of target for Cas9 cutting,
                        so a cut_site of 6 means that we start inserting
                        such that the inserted seq is 6 nucleotides from
                        the 3' end of the target
        """
        # The original barcode
        self.unedited_barcode = unedited_barcode
        self.orig_substr_lens = [len(s) for s in unedited_barcode]
        self.orig_length = sum(self.orig_substr_lens)
        self.num_targets = (len(self.unedited_barcode) - 1) // 2

        self.cut_sites = cut_sites
        # absolute positions of cut locations
        self.abs_cut_sites = [
            sum(self.orig_substr_lens[:2 * (i + 1)]) - cut_sites[i] for i in range(self.num_targets)
        ]

        # Range of positions for each target
        # regarding which positions must be unedited
        # for this target to still be active.
        self.pos_sites = []
        cumsum = np.cumsum(self.orig_substr_lens)
        for i in range(self.num_targets):
            right = cumsum[2 * (i + 1) - 1]
            left = right - self.orig_substr_lens[2 * (i + 1) - 1]
            self.pos_sites.append((left, right))

    def get_min_max_deact_targets(self, evt: Event):
        if evt.min_target > 1 and evt.start_pos < self.pos_sites[evt.min_target - 1][1]:
            min_deact_target = evt.min_target - 1
        else:
            min_deact_target = evt.min_target

        if evt.max_target <= self.num_targets - 1 and self.pos_sites[evt.max_target + 1][0] <= evt.del_end - 1:
            max_deact_target = evt.max_target + 1
        else:
            max_deact_target = evt.max_target

        return min_deact_target, max_deact_target
