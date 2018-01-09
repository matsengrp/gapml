from typing import List, Tuple

from indel_sets import TargetTract, AncState
from barcode_metadata import BarcodeMetadata

class StateSum:
    def __init__(self, tts_list: List[Tuple[TargetTract]] = [()]):
        self.tts_list = tts_list

    def add(self, tts: Tuple[TargetTract]):
        self.tts_list.append(tts)

    def __str__(self):
        return " OR ".join([str(tts) for tts in self.tts_list])
