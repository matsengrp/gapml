from typing import List, Tuple

class BarcodeMetadata:
    def __init__(self,
            num_targets: int,
            orig_sequence: str,
            cut_sites: List[int],
            pos_sites: List[Tuple[int, int]]):
        """
        @param pos_sites: essential positions that must be undisturbed for target to be active
                    pos(t) for t = 1:num_targets
        """
        self.num_targets = num_targets
        self.sequence = orig_sequence
        self.length = len(orig_sequence)
        self.cut_sites = cut_sites
        self.pos_sites = pos_sites
