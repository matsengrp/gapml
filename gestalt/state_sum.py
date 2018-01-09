from typing import List, Tuple

from indel_sets import TargetTract, AncState
from barcode_metadata import BarcodeMetadata

class StateSum:
    def __init__(self, tts_list: List[Tuple[TargetTract]] = []):
        self.tts_list = tts_list

def partition_tts(tts: Tuple[TargetTract], anc_state: AncState, bcode_metadata: BarcodeMetadata):
    if len(tts) == 0:
        return []

    tts_idx = 0
    for indel_set in anc_state.indel_set_list:
        cur_tt = tts[tts_idx]
