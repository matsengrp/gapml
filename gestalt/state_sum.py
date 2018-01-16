from typing import List, Tuple, Set

from indel_sets import TargetTract, AncState
from barcode_metadata import BarcodeMetadata

class StateSum:
    def __init__(self, tts_set: Set[Tuple[TargetTract]]):
        self.tts_set = tts_set

    def update(self, new_additions: Set[Tuple[TargetTract]]):
        self.tts_set.update(new_additions)

    def __str__(self):
        return " OR ".join([str(tts) for tts in self.tts_set])
